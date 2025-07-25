package api

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"path"
	"runtime"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/dnswlt/ogd-weather/service/api/internal/cache"
	"github.com/dnswlt/ogd-weather/service/api/internal/types"
	"github.com/dnswlt/ogd-weather/service/api/internal/ui"
	"github.com/dnswlt/ogd-weather/service/api/internal/version"
)

type Server struct {
	opts                ServerOptions
	chartServiceBaseURL *url.URL
	template            *template.Template
	lastReloadMu        sync.Mutex
	lastReload          time.Time
	client              *http.Client
	cache               cache.Cache
	startTime           time.Time
	bearerToken         string // Used for pages requiring simple Authorization
}

type ServerOptions struct {
	Addr                 string // Ex: "localhost:8080"
	ChartServiceEndpoint string // Ex: "http://localhost:8000"
	BaseDir              string // Directory under which "templates" and "static" dirs are expected
	DebugMode            bool
	LogRequests          bool
	CacheSize            int // Maximum approx. size for the response cache in bytes
}

func (s *Server) uptime() time.Duration {
	return time.Since(s.startTime)
}

func (s *Server) reloadTemplates() error {
	tmpl := template.New("root")
	tmpl = tmpl.Funcs(map[string]any{
		"datefmt":    ui.DateFmt,
		"wgs84tosvg": ui.WGS84ToSVG,
		"min2hours":  ui.MinutesToHours,
		"ms2kmh":     ui.MetersPerSecondToKilometersPerHour,
	})
	var err error
	s.template, err = tmpl.ParseGlob(path.Join(s.opts.BaseDir, "templates/*.html"))
	return err
}

type ServerOption func(s *Server) error

func WithBearerToken(token string) ServerOption {
	return func(s *Server) error {
		s.bearerToken = token
		return nil
	}
}

func NewServer(opts ServerOptions, moreOpts ...ServerOption) (*Server, error) {
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
		cache:               cache.New(opts.CacheSize),
		startTime:           time.Now(),
	}
	for _, o := range moreOpts {
		err := o(s)
		if err != nil {
			return nil, fmt.Errorf("error applying ServerOption: %v", err)
		}
	}

	if err := s.reloadTemplates(); err != nil {
		return nil, fmt.Errorf("failed to load templates: %w", err)
	}

	return s, nil
}

func (s *Server) serveHTMLPage(w http.ResponseWriter, r *http.Request, templateFile string) {
	var output bytes.Buffer
	// Pass on query parameters to the template.
	flatQuery := map[string]string{}
	q := r.URL.Query()
	for k := range q {
		flatQuery[k] = q.Get(k)
	}
	// Months and other periods (for dropdown)
	periods := ui.Periods(q.Get("period"))
	// Weather stations (dropdown)
	u := *s.chartServiceBaseURL
	u.Path = path.Join(u.Path, "/stations")
	stations, err := fetchBackendData[types.StationsResponse](r.Context(), s, u.String())
	if err != nil {
		log.Printf("Failed to fetch stations from backend: %v", err)
		http.Error(w, "backend error", http.StatusInternalServerError)
		return
	}

	err = s.template.ExecuteTemplate(&output, templateFile, map[string]any{
		"Query":    flatQuery,
		"Periods":  periods,
		"Stations": stations.Stations,
		"Selected": flatQuery["station"],
		"Nav": ui.NavBar(r.URL.Path,
			ui.Nav("/", "Trends"),
			ui.Nav("/sun_rain", "Sun & Rain"),
			ui.Nav("/day", "Daily"),
			ui.Nav("/map", "Map"),
		),
	})
	if err != nil {
		log.Printf("Failed to render template %q: %v", templateFile, err)
		http.Error(w, "Template rendering error", http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=UTF-8")
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

// withAuth is a middleware that checks for a valid Bearer token in the Authorization header.
func (s *Server) withAuth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if s.bearerToken == "" {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		authHeader := r.Header.Get("Authorization")

		var token string
		n, err := fmt.Sscanf(authHeader, "Bearer %s", &token)
		if err != nil || n != 1 {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		if token != s.bearerToken {
			http.Error(w, "Unauthorized", http.StatusUnauthorized)
			return
		}

		next.ServeHTTP(w, r)
	}
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

// canonicalURL returns a canonical form of the given rawURL.
// This is useful when URLs are used as cache keys.
func canonicalURL(rawURL string) (string, error) {
	u, err := url.Parse(rawURL)
	if err != nil {
		return "", err
	}

	q := u.Query()

	// sort repeated values for deterministic order
	for k := range q {
		sort.Strings(q[k])
	}
	u.RawQuery = q.Encode() // Encode sorts keys

	return u.String(), nil
}

func fetchBackendData[Response any](ctx context.Context, s *Server, backendURL string) (*Response, error) {
	// Canonicalize for stable cache keys
	if u, err := canonicalURL(backendURL); err == nil {
		backendURL = u
	}
	// Try cache first
	r, found := s.cache.Get(backendURL)
	if found {
		if resp, ok := r.(*Response); ok {
			return resp, nil
		}
	}
	// Not cached => fetch
	ctx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, backendURL, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request for %s: %v", backendURL, err)
	}
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("backend error for %s: %v", backendURL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, fmt.Errorf("backend returned status %s for %s.", resp.Status, backendURL)
	}
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %v", err)
	}
	var response Response
	if err := json.Unmarshal(body, &response); err != nil {
		return nil, fmt.Errorf("chart server returned invalid JSON (want %T): %v", response, err)
	}

	// Add to LRU cache, respecting backend's Cache-Control.
	policy, ttl := cache.ComputeCachePolicy(resp.Header, time.Now())
	switch policy {
	case cache.CachePolicyNoCache:
		// Never cache
	case cache.CachePolicyTTL:
		log.Printf("Adding %s to cache (ttl=%v)", backendURL, ttl)
		s.cache.Put(backendURL, &response, len(body), ttl)
	case cache.CachePolicyNone:
		// No cache policy specified by backend: opportunistically cache for 10min.
		ttl = 10 * time.Minute
		log.Printf("Adding %s to cache (default ttl=%v)", backendURL, ttl)
		s.cache.Put(backendURL, &response, len(body), ttl)
	}

	return &response, nil
}

func serveChartServiceURL[Response any](
	s *Server,
	w http.ResponseWriter,
	r *http.Request,
	templateName string,
	params map[string]any) {

	backendURL := s.chartServiceURL(r.URL).String()
	response, err := fetchBackendData[Response](r.Context(), s, backendURL)
	if err != nil {
		log.Printf("Error fetching backend data: %v", err)
		http.Error(w, "Backend error", http.StatusInternalServerError)
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

func (s *Server) serveStatus(w http.ResponseWriter) {
	type ServerStatus struct {
		StartTime           time.Time
		UptimeSeconds       float64
		CacheUsage          int
		CacheCapacity       int
		ChartServiceBaseURL string
		Options             ServerOptions
		GoVersion           string
		Hostname            string
		NumCPU              int
		BuildInfo           string
		HasBearerToken      bool
	}

	hostname, _ := os.Hostname()
	status := ServerStatus{
		StartTime:           s.startTime,
		UptimeSeconds:       s.uptime().Seconds(),
		CacheUsage:          s.cache.Usage(),
		CacheCapacity:       s.cache.Capacity(),
		ChartServiceBaseURL: s.chartServiceBaseURL.String(),
		Options:             s.opts,
		GoVersion:           runtime.Version(),
		Hostname:            hostname,
		NumCPU:              runtime.NumCPU(),
		BuildInfo:           version.FullVersion(),
		HasBearerToken:      s.bearerToken != "",
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Cache-Control", "no-cache")
	json.NewEncoder(w).Encode(status)
}

func (s *Server) Serve() error {
	mux := http.NewServeMux()

	// Root page
	mux.HandleFunc("GET /{$}", func(w http.ResponseWriter, r *http.Request) {
		s.serveHTMLPage(w, r, "index.html")
	})
	mux.HandleFunc("GET /day", func(w http.ResponseWriter, r *http.Request) {
		s.serveHTMLPage(w, r, "day.html")
	})
	mux.HandleFunc("GET /sun_rain", func(w http.ResponseWriter, r *http.Request) {
		s.serveHTMLPage(w, r, "sun_rain.html")
	})
	mux.HandleFunc("GET /map", func(w http.ResponseWriter, r *http.Request) {
		s.serveHTMLPage(w, r, "map.html")
	})
	mux.Handle("DELETE /admin/cache/responses", s.withAuth(func(w http.ResponseWriter, r *http.Request) {
		s.cache.Purge()
		w.Header().Set("Content-Type", "text/plain; charset=UTF-8")
		w.Write([]byte("OK\n"))
	}))

	// Static resources (JavaScript, CSS, etc.)
	staticDir := path.Join(s.opts.BaseDir, "static")
	mux.Handle("GET /static/", http.StripPrefix("/static/",
		http.FileServer(http.Dir(staticDir))))

	// Health check. Useful for cloud deployments.
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, "OK")
	})
	// Status page. Always a treasure in Cloud deployments.
	mux.HandleFunc("/status", func(w http.ResponseWriter, r *http.Request) {
		s.serveStatus(w)
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
				serveChartServiceURL[types.StationSummaryResponse](s, w, r, "station_summary.html", nil)
				return
			}
			proxy.ServeHTTP(w, r)
		})
	mux.HandleFunc("GET /stations/{stationID}/info",
		func(w http.ResponseWriter, r *http.Request) {
			if acceptsHTML(r.Header.Get("Accept")) {
				serveChartServiceURL[types.StationInfoResponse](s, w, r, "station_info.html", nil)
				return
			}
			proxy.ServeHTTP(w, r)
		})
	mux.HandleFunc("GET /stations/{stationID}/daily",
		func(w http.ResponseWriter, r *http.Request) {
			if acceptsHTML(r.Header.Get("Accept")) {
				serveChartServiceURL[types.StationMeasurementsResponse](s, w, r, "daily_measurements.html", nil)
				return
			}
			proxy.ServeHTTP(w, r)
		})
	mux.HandleFunc("GET /stations",
		func(w http.ResponseWriter, r *http.Request) {
			accept := r.Header.Get("Accept")
			if acceptsHTML(accept) {
				serveChartServiceURL[types.StationsResponse](s, w, r, "station_options.html",
					map[string]any{
						"SelectedStation": r.URL.Query().Get("station"),
					})
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
