package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"html/template"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"path"
	"strings"
	"sync"
	"time"

	"github.com/dnswlt/ogd-weather/service/api/internal/types"
	"github.com/dnswlt/ogd-weather/service/api/internal/ui"
)

type Server struct {
	opts                ServerOptions
	chartServiceBaseURL *url.URL
	template            *template.Template
	lastReloadMu        sync.Mutex
	lastReload          time.Time
	client              *http.Client
}

type ServerOptions struct {
	Addr                 string // Ex: "localhost:8080"
	ChartServiceEndpoint string // Ex: "http://localhost:8000"
	TemplateDir          string // Ex: "./templates"
	DebugMode            bool
	LogRequests          bool
}

func (s *Server) reloadTemplates() error {
	tmpl := template.New("root")
	tmpl = tmpl.Funcs(map[string]any{
		"datefmt": ui.DateFmt,
	})
	var err error
	s.template, err = tmpl.ParseGlob(path.Join(s.opts.TemplateDir, "*.html"))
	return err
}

func NewServer(opts ServerOptions) (*Server, error) {
	chartServiceBaseURL, err := url.Parse(opts.ChartServiceEndpoint)
	if err != nil {
		return nil, fmt.Errorf("invalid chartServiceEndpoint URL: %v", err)
	}
	client := &http.Client{
		Transport: &http.Transport{
			Proxy: http.ProxyFromEnvironment,
			DialContext: (&net.Dialer{
				Timeout:   5 * time.Second,  // TCP connect timeout
				KeepAlive: 15 * time.Second, // keep-alive for connection reuse
			}).DialContext,
			MaxIdleConns:        100,
			IdleConnTimeout:     120 * time.Second,
			TLSHandshakeTimeout: 5 * time.Second,
		},
	}
	s := &Server{
		opts:                opts,
		chartServiceBaseURL: chartServiceBaseURL,
		client:              client,
	}

	if err := s.reloadTemplates(); err != nil {
		return nil, fmt.Errorf("failed to load templates: %w", err)
	}

	return s, nil
}

func (s *Server) serveHTML(w http.ResponseWriter, templateFile string) {
	var output bytes.Buffer
	err := s.template.ExecuteTemplate(&output, templateFile, nil)
	if err != nil {
		log.Printf("Failed to render template %q: %v", templateFile, err)
		http.Error(w, "Template rendering error", http.StatusInternalServerError)
		return
	}
	w.Write(output.Bytes())
}

// withRequestLogging wraps a handler and logs each request if in debug mode.
// Logs include method, path, remote address, and duration.
func (s *Server) withRequestLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()

		// Wrap ResponseWriter to capture status code
		lrw := &loggingResponseWriter{ResponseWriter: w}

		next.ServeHTTP(lrw, r)

		duration := time.Since(start)
		log.Printf("%s %s %d %dms (remote=%s)",
			r.Method,
			r.URL.Path,
			lrw.statusCode,
			duration.Milliseconds(),
			r.RemoteAddr,
		)
	})
}

type loggingResponseWriter struct {
	http.ResponseWriter
	statusCode int
}

func (lrw *loggingResponseWriter) WriteHeader(code int) {
	lrw.statusCode = code
	lrw.ResponseWriter.WriteHeader(code)
}

func (lrw *loggingResponseWriter) Write(b []byte) (int, error) {
	if lrw.statusCode == 0 { // no explicit status yet => implies 200
		lrw.WriteHeader(http.StatusOK)
	}
	return lrw.ResponseWriter.Write(b)
}

// withTemplateReload wraps a handler and reloads templates in debug mode,
// but at most once every second (1s). The first triggering request will block
// until reload completes.
func (s *Server) withTemplateReload(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if s.opts.DebugMode {
			now := time.Now()

			func() {
				s.lastReloadMu.Lock()
				defer s.lastReloadMu.Unlock()

				if now.Sub(s.lastReload) > 1*time.Second {
					if err := s.reloadTemplates(); err != nil {
						log.Printf("Failed to reload templates: %v", err)
						http.Error(w, "Internal Server Error", http.StatusInternalServerError)
						return
					}
					s.lastReload = now
				}
			}()
		}

		next.ServeHTTP(w, r)
	})
}

func (s *Server) Serve() error {
	mux := http.NewServeMux()

	// Root page
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		s.serveHTML(w, "index.html")
	})
	mux.HandleFunc("GET /day", func(w http.ResponseWriter, r *http.Request) {
		s.serveHTML(w, "day.html")
	})
	mux.HandleFunc("GET /sun_rain", func(w http.ResponseWriter, r *http.Request) {
		s.serveHTML(w, "sun_rain.html")
	})

	// Health check. Useful for cloud deployments.
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, "OK")
	})

	// Reverse proxy
	proxy := httputil.NewSingleHostReverseProxy(s.chartServiceBaseURL)
	mux.HandleFunc("GET /stations/{stationID}/charts/{chartType}",
		func(w http.ResponseWriter, r *http.Request) {
			if acceptsHTML(r.Header.Get("Accept")) {
				s.serveStationsChartSnippet(w, r)
				return
			}
			proxy.ServeHTTP(w, r)
		})
	mux.HandleFunc("GET /stations/{stationID}/summary",
		func(w http.ResponseWriter, r *http.Request) {
			if acceptsHTML(r.Header.Get("Accept")) {
				s.serveSummarySnippet(w, r)
				return
			}
			proxy.ServeHTTP(w, r)
		})
	mux.HandleFunc("GET /stations/{stationID}/daily",
		func(w http.ResponseWriter, r *http.Request) {
			if acceptsHTML(r.Header.Get("Accept")) {
				s.serveDailySnippet(w, r)
				return
			}
			proxy.ServeHTTP(w, r)
		})
	mux.HandleFunc("GET /stations",
		func(w http.ResponseWriter, r *http.Request) {
			accept := r.Header.Get("Accept")
			if acceptsHTML(accept) {
				s.serveStationsSnippet(w, r)
				return
			}
			proxy.ServeHTTP(w, r)
		})

	var handler http.Handler = mux
	if s.opts.DebugMode {
		handler = s.withTemplateReload(handler)
	}
	if s.opts.LogRequests {
		handler = s.withRequestLogging(handler)
	}

	modeInfo := "(prod mode)"
	if s.opts.DebugMode {
		modeInfo = "(debug mode)"
	}
	log.Printf("Go API server %s on http://%s", modeInfo, s.opts.Addr)
	return http.ListenAndServe(s.opts.Addr, handler)
}

func acceptsHTML(accept string) bool {
	return strings.Contains(accept, "text/html") ||
		strings.Contains(accept, "application/xhtml+xml") ||
		strings.Contains(accept, "*/*")
}

// chartServiceURL creates the forwarding URL to the chart service endpoint
// for the given original URL targeting this server.
func (s *Server) chartServiceURL(originalURL *url.URL) *url.URL {
	u := *s.chartServiceBaseURL
	u.Path = originalURL.Path
	u.RawQuery = originalURL.RawQuery
	return &u
}

func serveChartServiceURL[Response any](
	s *Server,
	w http.ResponseWriter,
	r *http.Request,
	templateName string,
	params map[string]any) {

	u := s.chartServiceURL(r.URL)

	ctx, cancel := context.WithTimeout(r.Context(), 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, u.String(), nil)
	if err != nil {
		log.Printf("Failed to create request for %s: %v", u.String(), err)
		http.Error(w, "Backend error", http.StatusInternalServerError)
		return
	}
	resp, err := s.client.Do(req)
	if err != nil {
		log.Printf("Backend error for %s: %v", u.String(), err)
		http.Error(w, "Backend error", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		log.Printf("Backend returned status %s for %s.", resp.Status, u.String())
		http.Error(w, "", resp.StatusCode)
		return
	}

	var response Response
	err = json.NewDecoder(resp.Body).Decode(&response)
	if err != nil {
		log.Printf("Chart server returned invalid JSON (want %T): %v", response, err)
		http.Error(w, "Read backend error", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "text/html")

	var output bytes.Buffer
	templateData := map[string]any{
		"Response": response,
	}
	for k, v := range params {
		templateData[k] = v
	}
	err = s.template.ExecuteTemplate(&output, templateName, templateData)
	if err != nil {
		log.Printf("Failed to render template %q: %v", templateName, err)
		http.Error(w, "Template rendering error", http.StatusInternalServerError)
		return
	}
	w.Write(output.Bytes())
}

func (s *Server) serveStationsChartSnippet(w http.ResponseWriter, r *http.Request) {
	chartType := r.PathValue("chartType")
	serveChartServiceURL[types.VegaSpecResponse](s, w, r, "vega_embed.html", map[string]any{
		"ChartType": chartType,
	})
}

func (s *Server) serveSummarySnippet(w http.ResponseWriter, r *http.Request) {
	serveChartServiceURL[types.StationSummaryResponse](s, w, r, "station_summary.html", nil)
}

func (s *Server) serveDailySnippet(w http.ResponseWriter, r *http.Request) {
	serveChartServiceURL[types.StationMeasurementsResponse](s, w, r, "daily_measurements.html", nil)
}

func (s *Server) serveStationsSnippet(w http.ResponseWriter, r *http.Request) {
	serveChartServiceURL[types.StationsResponse](s, w, r, "station_options.html", nil)
}
