package api

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
)

type Server struct {
	addr                 string
	chartServiceEndpoint *url.URL
}

func NewServer(addr, chartServiceEndpoint string) (*Server, error) {
	endpoint, err := url.Parse(chartServiceEndpoint)
	if err != nil {
		return nil, fmt.Errorf("invalid chartServiceEndpoint URL: %v", err)
	}
	return &Server{
		addr:                 addr,
		chartServiceEndpoint: endpoint,
	}, nil
}

func (s *Server) Serve() error {
	mux := http.NewServeMux()

	// Static files
	mux.Handle("/", http.FileServer(http.Dir("static")))

	// Reverse proxy
	proxy := httputil.NewSingleHostReverseProxy(s.chartServiceEndpoint)
	mux.Handle("GET /stations/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if acceptsHTML(r.Header.Get("Accept")) {
			s.serveChartSnippet(w, r)
			return
		}
		proxy.ServeHTTP(w, r)
	}))
	mux.HandleFunc("GET /stations", func(w http.ResponseWriter, r *http.Request) {
		accept := r.Header.Get("Accept")
		if acceptsHTML(accept) {
			// Serve <option> elements for htmx
			s.serveStationOptions(w, r)
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

func (s *Server) serveChartSnippet(w http.ResponseWriter, r *http.Request) {
	// Call the Python backend to get the JSON Vega spec
	resp, err := http.Get(s.chartServiceEndpoint.String() + r.URL.Path)
	if err != nil {
		http.Error(w, "Backend error", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 400 {
		http.Error(w, "", resp.StatusCode)
	}

	spec, err := io.ReadAll(resp.Body)
	if err != nil {
		http.Error(w, "Read backend error", http.StatusInternalServerError)
		return
	}

	// Return <script> for htmx
	w.Header().Set("Content-Type", "text/html")
	io.WriteString(w, `<script id="chart-script">
        vegaEmbed("#chart", `)
	w.Write(spec)
	io.WriteString(w, `, {actions:false});
    </script>`)
}

func (s *Server) serveStationOptions(w http.ResponseWriter, r *http.Request) {
	resp, err := http.Get(s.chartServiceEndpoint.String() + r.URL.Path)
	if err != nil {
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
	var data StationsResponse
	if err := json.Unmarshal(body, &data); err != nil {
		http.Error(w, "Parse backend JSON error", http.StatusInternalServerError)
		return
	}

	// Return <option> elements
	w.Header().Set("Content-Type", "text/html")
	for _, st := range data.Stations {
		fmt.Fprintf(w, `<option value="%s">%s %s (%s)</option>`,
			st.Abbr, st.Name, st.Canton, st.Abbr)
	}
}
