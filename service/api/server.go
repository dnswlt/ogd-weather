package api

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
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
	"path/filepath"
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
	hashedFilenames     map[string]string
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

// assetHash returns the hashed filename for the given hrefPath.
// Example: "/static/dist/main.js" => "/static/dist/main.<hash>.js".
// This is used for Vite-generated assets which include the build hash
// in their filename for cache busting.
// If no Vite manifest.json was scanned at server startup, hrefPath is returned unchanged.
func (s *Server) assetHash(hrefPath string) (string, error) {
	if !strings.HasPrefix(hrefPath, "/static/dist/") {
		return "", fmt.Errorf("assetHash must only be used for /static/dist/ resources")
	}
	if strings.Contains(hrefPath, "..") {
		return "", fmt.Errorf("path must not contain \"..\"")
	}
	// Not running with Vite config => return hrefPath unchanged.
	if s.hashedFilenames == nil {
		return hrefPath, nil
	}

	// Split "/static/dist/main.js" into ("/static/dist/", "main.js"):
	hrefDir, name := path.Split(hrefPath)
	if hash, ok := s.hashedFilenames[name]; ok {
		return hrefDir + hash, nil
	}
	// no hashed version known
	return hrefPath, nil
}

func (s *Server) reloadTemplates() error {
	tmpl := template.New("root")
	tmpl = tmpl.Funcs(ui.AllFuncs)
	tmpl = tmpl.Funcs(map[string]any{
		"assetHash": s.assetHash,
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

func WithViteManifest() ServerOption {
	unhashed := func(file string) string {
		parts := strings.Split(file, ".")
		// A valid Vite filename will have at least 3 parts: "base", "hash", "extension".
		if len(parts) < 3 {
			return file
		}
		// Remove the second-to-last element (the hash)
		hashIndex := len(parts) - 2
		parts = append(parts[:hashIndex], parts[hashIndex+1:]...)
		return strings.Join(parts, ".")
	}
	return func(s *Server) error {
		manifestFile := filepath.Join(s.opts.BaseDir, "static", "dist", ".vite", "manifest.json")
		f, err := os.Open(manifestFile)
		if err != nil {
			return fmt.Errorf("no Vite manifest.json found: %v", err)
		}
		var manifest map[string]struct {
			File string
			CSS  []string
		}
		if err := json.NewDecoder(f).Decode(&manifest); err != nil {
			return fmt.Errorf("could not parse manifest.json: %v", err)
		}
		hashedFiles := make(map[string]string)
		for _, info := range manifest {
			// .js
			hashedFiles[unhashed(info.File)] = info.File
			// .css
			for _, css := range info.CSS {
				hashedFiles[unhashed(css)] = css
			}
		}
		log.Printf("Processed %d files in Vite manifest %s", len(hashedFiles), manifestFile)
		s.hashedFilenames = hashedFiles
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

func statusCodeFromError(err error) int {
	var be *BackendError
	if errors.As(err, &be) {
		return be.StatusCode()
	}
	return http.StatusInternalServerError
}

func (s *Server) backendError(w http.ResponseWriter, err error, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	log.Printf("%s: %v", msg, err)
	statusCode := statusCodeFromError(err)
	http.Error(w, fmt.Sprintf("%d %s", statusCode, http.StatusText(statusCode)), statusCode)
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
		s.backendError(w, err, "Failed to fetch stations from backend")
		return
	}

	nav := ui.NewNavBar(
		ui.NavItem("/ui/timeline", "Trends").Params("station", "from_year", "to_year", "period", "window"),
		ui.NavItem("/ui/sun_rain", "Sun & Rain").Params("station", "from_year", "to_year", "period"),
		ui.NavItem("/ui/day", "Day").Params("station", "date"),
		ui.NavItem("/ui/year", "Year").Params("station", "year"),
		ui.NavItem("/ui/map", "Map").Params("station"),
	).SetActive(r.URL.Path).SetParams(q)

	err = s.template.ExecuteTemplate(&output, templateFile, map[string]any{
		"Query":           flatQuery,
		"Periods":         periods,
		"Stations":        stations.Stations,
		"SelectedStation": flatQuery["station"],
		"NavBar":          nav,
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

func (s *Server) withCacheControl(h http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Never cache our own files in /static/dist.
		// When using StripPrefix("/static/", ...), the path appears as "dist/"here.
		if strings.HasPrefix(r.URL.Path, "dist/") || strings.HasPrefix(r.URL.Path, "/static/dist/") {
			h.ServeHTTP(w, r)
			return
		}

		// Set a max-age of 1 year for static resources that do not change.
		// We use version numbers in file names for cache busting.
		ext := path.Ext(r.URL.Path)
		switch ext {
		case ".js", ".css", ".svg", ".png", ".jpg", ".jpeg", ".ico", ".map",
			".woff", ".woff2", ".ttf", ".eot", ".otf",
			".webp", ".avif":
			w.Header().Set("Cache-Control", "public, max-age=31536000, immutable")
		}
		h.ServeHTTP(w, r)
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

func (s *Server) newChartServiceURL(path string) *url.URL {
	u := *s.chartServiceBaseURL
	u.Path = path
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
	// Ensure we get JSON back. The Python backend might also support
	// text/html, e.g. for debugging.
	req.Header.Add("Accept", "application/json")
	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("backend error for %s: %v", backendURL, err)
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		return nil, NewBackendError(backendURL, resp.StatusCode)
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
		s.backendError(w, err, "Failed to fetch data (%T) from backend", response)
		return
	}

	w.Header().Set("Content-Type", "text/html; charset=UTF-8")

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
	if chartType == "" {
		log.Printf("Missing {chartType} in URL path for %s", r.URL.Path)
		http.Error(w, "Missing chartType in URL path", http.StatusInternalServerError)
		return
	}
	serveChartServiceURL[types.VegaSpecResponse](s, w, r, "vega_embed.html", map[string]any{
		"ChartType": chartType,
	})
}

func (s *Server) serveStatus(w http.ResponseWriter, r *http.Request) {

	var backendStatus map[string]any
	q := r.URL.Query()
	if q.Get("format") == "full" {
		u := s.newChartServiceURL("/status")
		status, err := fetchBackendData[map[string]any](r.Context(), s, u.String())
		if err != nil {
			log.Printf("Error fetching /status from backend: %v", err)

		}
		backendStatus = *status
	}
	hostname, _ := os.Hostname()

	status := types.ServerStatus{
		StartTime:           s.startTime,
		UptimeSeconds:       s.uptime().Seconds(),
		CacheUsage:          s.cache.Usage(),
		CacheCapacity:       s.cache.Capacity(),
		ChartServiceBaseURL: s.chartServiceBaseURL.String(),
		Options: types.ServerOptions{
			Addr:                 s.opts.Addr,
			ChartServiceEndpoint: s.opts.ChartServiceEndpoint,
			DebugMode:            s.opts.DebugMode,
			CacheSize:            ui.FormatBytesIEC(int64(s.opts.CacheSize)),
		},
		GoVersion:      runtime.Version(),
		Hostname:       hostname,
		NumCPU:         runtime.NumCPU(),
		BuildInfo:      version.FullVersion(),
		HasBearerToken: s.bearerToken != "",
		BackendStatus:  backendStatus,
	}

	w.Header().Set("Content-Type", "application/json; charset=UTF-8")
	w.Header().Set("Cache-Control", "no-cache")
	json.NewEncoder(w).Encode(status)
}

func withQueryParams(u *url.URL, params map[string]string) *url.URL {
	newURL := *u
	query := newURL.Query()
	for k, v := range params {
		query.Add(k, v)
	}
	newURL.RawQuery = query.Encode()
	return &newURL
}

func (s *Server) Serve() error {
	mux := http.NewServeMux()

	// Root UI page
	mux.HandleFunc("GET /ui", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/ui/timeline", http.StatusTemporaryRedirect)
	})
	mux.HandleFunc("GET /ui/timeline", func(w http.ResponseWriter, r *http.Request) {
		s.serveHTMLPage(w, r, "timeline.html")
	})
	mux.HandleFunc("GET /ui/day", func(w http.ResponseWriter, r *http.Request) {
		// Redirect to 2daysago if date= query param is missing.
		if !r.URL.Query().Has("date") {
			newURL := withQueryParams(r.URL, map[string]string{
				"date": time.Now().AddDate(0, 0, -2).Format("2006-01-02"),
			})
			http.Redirect(w, r, newURL.RequestURI(), http.StatusFound)
		}
		s.serveHTMLPage(w, r, "day.html")
	})
	mux.HandleFunc("GET /ui/year", func(w http.ResponseWriter, r *http.Request) {
		// Redirect to the year from N days ago if year= query param is missing.
		if !r.URL.Query().Has("year") {
			newURL := withQueryParams(r.URL, map[string]string{
				"year": time.Now().AddDate(0, 0, -7).Format("2006"),
			})
			http.Redirect(w, r, newURL.RequestURI(), http.StatusFound)
		}
		s.serveHTMLPage(w, r, "year.html")
	})
	mux.HandleFunc("GET /ui/sun_rain", func(w http.ResponseWriter, r *http.Request) {
		s.serveHTMLPage(w, r, "sun_rain.html")
	})
	mux.HandleFunc("GET /ui/map", func(w http.ResponseWriter, r *http.Request) {
		s.serveHTMLPage(w, r, "map.html")
	})
	mux.Handle("DELETE /admin/cache/responses", s.withAuth(func(w http.ResponseWriter, r *http.Request) {
		s.cache.Purge()
		w.Header().Set("Content-Type", "text/plain; charset=UTF-8")
		w.Write([]byte("OK\n"))
	}))

	// Static resources (JavaScript, CSS, etc.)
	staticDir := path.Join(s.opts.BaseDir, "static")
	mux.Handle("GET /static/",
		http.StripPrefix("/static/",
			s.withCacheControl(
				http.FileServer(http.Dir(staticDir)),
			),
		),
	)

	// Health check. Useful for cloud deployments.
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		fmt.Fprintln(w, "OK")
	})
	// Status page. Always a treasure in Cloud deployments.
	mux.HandleFunc("/status", func(w http.ResponseWriter, r *http.Request) {
		s.serveStatus(w, r)
	})

	// Reverse proxy
	proxy := httputil.NewSingleHostReverseProxy(s.chartServiceBaseURL)
	mux.HandleFunc("GET /stations/{stationID}/charts/{periodName}/{chartType}",
		func(w http.ResponseWriter, r *http.Request) {
			if acceptsHTML(r.Header.Get("Accept")) {
				s.serveStationsChartSnippet(w, r)
				return
			}
			proxy.ServeHTTP(w, r)
		})
	mux.HandleFunc("GET /stations/{stationID}/charts/{periodName}/{date}/{chartType}",
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
	mux.HandleFunc("GET /stations/{stationID}/stats/day/{date}/measurements",
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
	mux.HandleFunc("GET /stations/{station_abbr}/data/",
		func(w http.ResponseWriter, r *http.Request) {
			proxy.ServeHTTP(w, r)
		})

	// Default route (all other paths): redirect to the UI home page
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "", http.StatusBadRequest)
			return
		}
		if r.Header.Get("Hx-Request") != "" {
			// Do not redirect htmx requests, those should only request valid paths.
			http.Error(w, "", http.StatusNotFound)
			return
		}
		refererURL, err := url.Parse(r.Header.Get("Referer"))
		if err == nil && refererURL.Host == r.Host {
			// Request is coming from our own domain: this indicates an internal broken link.
			http.Error(w, "Broken link", http.StatusNotFound)
			return
		}
		// Redirect GET to the UI home page.
		http.Redirect(w, r, "/ui", http.StatusTemporaryRedirect)
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
