package api

import (
	"bytes"
	"encoding/json"
	"fmt"
	"html/template"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"path"
	"strings"

	"github.com/dnswlt/ogd-weather/service/api/internal/types"
	"github.com/dnswlt/ogd-weather/service/api/internal/ui"
)

type Server struct {
	addr                 string
	chartServiceEndpoint *url.URL
	template             *template.Template
}

type ServerOptions struct {
	Addr                 string // Ex: "localhost:8080"
	ChartServiceEndpoint string // Ex: "http://localhost:8000"
	TemplateDir          string // Ex: "./templates"
}

func NewServer(opts ServerOptions) (*Server, error) {
	endpoint, err := url.Parse(opts.ChartServiceEndpoint)
	if err != nil {
		return nil, fmt.Errorf("invalid chartServiceEndpoint URL: %v", err)
	}
	tmpl := template.New("root")
	tmpl = tmpl.Funcs(map[string]any{
		"datefmt": ui.DateFmt,
	})
	tmpl, err = tmpl.ParseGlob(path.Join(opts.TemplateDir, "*.html"))

	if err != nil {
		return nil, fmt.Errorf("failed to read templates: %w", err)
	}
	return &Server{
		addr:                 opts.Addr,
		chartServiceEndpoint: endpoint,
		template:             tmpl,
	}, nil
}

func (s *Server) Serve() error {
	mux := http.NewServeMux()

	// Static files
	mux.Handle("/", http.FileServer(http.Dir("static")))

	// Reverse proxy
	proxy := httputil.NewSingleHostReverseProxy(s.chartServiceEndpoint)
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
	mux.HandleFunc("GET /stations",
		func(w http.ResponseWriter, r *http.Request) {
			accept := r.Header.Get("Accept")
			if acceptsHTML(accept) {
				// Serve <option> elements for htmx
				s.serveStationsSnippet(w, r)
				return
			}
			// Otherwise just proxy JSON
			proxy.ServeHTTP(w, r)
		})

	log.Printf("Go API server on http://%s", s.addr)
	return http.ListenAndServe(s.addr, mux)
}

func acceptsHTML(accept string) bool {
	return strings.Contains(accept, "text/html") ||
		strings.Contains(accept, "application/xhtml+xml") ||
		strings.Contains(accept, "*/*")
}

// chartServiceURL creates the forwarding URL to the chart service endpoint
// for the given original URL targeting this server.
func (s *Server) chartServiceURL(originalURL *url.URL) *url.URL {
	u := *s.chartServiceEndpoint
	u.Path = originalURL.Path
	u.RawQuery = originalURL.RawQuery
	return &u
}

func (s *Server) serveStationsChartSnippet(w http.ResponseWriter, r *http.Request) {
	chartType := r.PathValue("chartType")

	// Call the Python backend to get the JSON Vega spec
	u := s.chartServiceURL(r.URL)
	resp, err := http.Get(u.String())
	if err != nil {
		log.Printf("Backend error for %s: %v", r.URL.Path, err)
		http.Error(w, "Backend error", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		http.Error(w, "", resp.StatusCode)
		return
	}

	// Parse Vega spec as JSON to ensure it's valid, and to safely render
	// it in the html template.
	var vegaSpec map[string]any
	err = json.NewDecoder(resp.Body).Decode(&vegaSpec)
	if err != nil {
		log.Printf("Chart server returned invalid JSON: %v", err)
		http.Error(w, "Read backend error", http.StatusInternalServerError)
		return
	}

	// Return <script> for htmx
	w.Header().Set("Content-Type", "text/html")

	var output bytes.Buffer
	err = s.template.ExecuteTemplate(&output, "vega_embed.html", map[string]any{
		"ChartType": chartType,
		"VegaSpec":  vegaSpec,
	})
	if err != nil {
		log.Printf("Failed to render template: %v", err)
		http.Error(w, "Template rendering error", http.StatusInternalServerError)
		return
	}
	w.Write(output.Bytes())
}

func serveChartServiceURL[Response any](
	s *Server,
	w http.ResponseWriter,
	r *http.Request,
	templateName string,
	params map[string]any) {

	u := s.chartServiceURL(r.URL)
	resp, err := http.Get(u.String())
	if err != nil {
		log.Printf("Backend error for %s: %v", r.URL.Path, err)
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

func (s *Server) serveSummarySnippet(w http.ResponseWriter, r *http.Request) {
	serveChartServiceURL[types.StationSummaryResponse](s, w, r, "station_summary.html", nil)
}

func (s *Server) serveStationsSnippet(w http.ResponseWriter, r *http.Request) {
	resp, err := http.Get(s.chartServiceEndpoint.String() + r.URL.Path)
	if err != nil {
		log.Printf("Backend error for %s: %v", r.URL.Path, err)
		http.Error(w, "Backend error", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "Read backend error", http.StatusInternalServerError)
		return
	}

	// Parse JSON
	var data types.StationsResponse
	if err := json.Unmarshal(body, &data); err != nil {
		log.Printf("Error parsing stations JSON: %v", err)
		http.Error(w, "Parse backend JSON error", http.StatusInternalServerError)
		return
	}

	// Return <option> elements
	w.Header().Set("Content-Type", "text/html")
	var output bytes.Buffer
	err = s.template.ExecuteTemplate(&output, "station_options.html", map[string]any{
		"Stations": data.Stations,
	})
	if err != nil {
		log.Printf("Failed to render template: %v", err)
		http.Error(w, "Template rendering error", http.StatusInternalServerError)
		return
	}
	w.Write(output.Bytes())
}
