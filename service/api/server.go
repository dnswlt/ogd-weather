package api

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
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
	mux.Handle("/stations/", http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if acceptsHTML(r.Header.Get("Accept")) {
			s.serveChartSnippet(w, r)
			return
		}
		proxy.ServeHTTP(w, r)
	}))

	log.Printf("Go API server on http://%s", s.addr)
	return http.ListenAndServe(s.addr, mux)
}

func acceptsHTML(accept string) bool {
	return accept == "text/html" || accept == "application/xhtml+xml" || accept == "*/*"
}

func (s *Server) serveChartSnippet(w http.ResponseWriter, r *http.Request) {
	// Call the Python backend to get the JSON spec
	resp, err := http.Get(s.chartServiceEndpoint.String() + r.URL.Path)
	if err != nil {
		http.Error(w, "Backend error", http.StatusInternalServerError)
		return
	}
	defer resp.Body.Close()

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
