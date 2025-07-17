package main

import (
	"flag"
	"log"

	"github.com/dnswlt/ogd-weather/service/api"
)

func main() {
	var opts api.ServerOptions
	flag.StringVar(&opts.Addr, "addr", "127.0.0.1:8080", "Address that the server listens on.")
	flag.StringVar(&opts.ChartServiceEndpoint, "chart-service-endpoint", "http://127.0.0.1:8000", "URL of the chart service backend.")
	flag.StringVar(&opts.TemplateDir, "template-dir", "./templates", "Directory containing server templates.")
	flag.Parse()

	server, err := api.NewServer(opts)
	if err != nil {
		log.Fatalf("Error creating server: %v", err)

	}
	log.Fatal(server.Serve())
}
