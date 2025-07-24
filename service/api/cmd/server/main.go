package main

import (
	"flag"
	"log"
	"os"

	"github.com/dnswlt/ogd-weather/service/api"
)

func main() {
	// Get defaults from env, if set.
	chartServiceEndpoint := os.Getenv("OGD_CHART_SERVICE_ENDPOINT")
	if chartServiceEndpoint == "" {
		chartServiceEndpoint = "http://127.0.0.1:8000" // local default
	}
	addr := os.Getenv("OGD_API_ADDR")
	if addr == "" {
		addr = "127.0.0.1:8080" // local default
	}

	var opts api.ServerOptions
	flag.StringVar(&opts.Addr, "addr", addr, "Address that the server listens on.")
	flag.StringVar(&opts.ChartServiceEndpoint, "chart-service-endpoint", chartServiceEndpoint, "URL of the chart service backend.")
	flag.StringVar(&opts.TemplateDir, "template-dir", "./templates", "Directory containing server templates.")
	flag.BoolVar(&opts.DebugMode, "debug", false, "If specified, the server runs in debug mode.")
	flag.BoolVar(&opts.LogRequests, "log-requests", false, "If true, the server write request logs.")
	flag.IntVar(&opts.CacheSize, "cache-size", 10<<20, "Approximate size of the response cache in bytes.")
	flag.Parse()

	server, err := api.NewServer(opts)
	if err != nil {
		log.Fatalf("Error creating server: %v", err)

	}
	log.Fatal(server.Serve())
}
