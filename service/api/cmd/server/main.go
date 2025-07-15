package main

import (
	"log"

	"github.com/dnswlt/ogd-weather/service/api"
)

func main() {
	server, err := api.NewServer("127.0.0.1:8080", "http://127.0.0.1:8000")
	if err != nil {
		log.Fatalf("Error creating server: %v", err)

	}
	log.Fatal(server.Serve())
}
