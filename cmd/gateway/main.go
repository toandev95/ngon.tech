package main

import (
	"context"
	"errors"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"ngon.tech/internal/config"
	"ngon.tech/internal/httpapi"
)

func main() {
	defaultPath := os.Getenv("CONFIG_FILE")
	if defaultPath == "" {
		executable, err := os.Executable()
		if err != nil {
			log.Fatalf("resolve executable path: %v", err)
		}
		defaultPath = filepath.Join(filepath.Dir(executable), "config.toml")
	}
	path := flag.String("config", defaultPath, "TOML config file")
	flag.Parse()
	cfg, err := config.Load(*path)
	if err != nil {
		log.Fatal(err)
	}

	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	server := &http.Server{
		Addr:              cfg.Server.Address,
		Handler:           httpapi.NewRouter(cfg),
		ReadHeaderTimeout: 10 * time.Second,
		IdleTimeout:       10 * time.Minute,
	}
	done := make(chan struct{})
	go func() {
		defer close(done)
		<-ctx.Done()
		stop()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		if err := server.Shutdown(shutdownCtx); err != nil {
			log.Printf("shutdown: %v", err)
			if err := server.Close(); err != nil {
				log.Printf("close server: %v", err)
			}
		}
	}()

	log.Printf("gateway listening on %s", server.Addr)
	if err := server.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
		log.Fatal(err)
	}
	<-done
}
