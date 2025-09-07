// Copyright 2025 The Go MCP SDK Authors. All rights reserved.
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

var (
	host  = flag.String("host", "localhost", "host to connect to/listen on")
	port  = flag.Int("port", 8000, "port number to connect to/listen on")
	proto = flag.String("proto", "http", "if set, use as proto:// part of URL (ignored for server)")
)

func main() {
	out := flag.CommandLine.Output()
	flag.Usage = func() {
		fmt.Fprintf(out, "Usage: %s <client|server> [-proto <http|https>] [-port <port] [-host <host>]\n\n", os.Args[0])
		fmt.Fprintf(out, "This program demonstrates MCP over HTTP using the streamable transport.\n")
		fmt.Fprintf(out, "It can run as either a server or client.\n\n")
		fmt.Fprintf(out, "Options:\n")
		flag.PrintDefaults()
		fmt.Fprintf(out, "\nExamples:\n")
		fmt.Fprintf(out, "  Custom host/port: %s -port 9000 -host 0.0.0.0 server\n", os.Args[0])
		os.Exit(1)
	}
	flag.Parse()

	if *proto != "http" {
		log.Fatalf("Server only works with 'http' (you passed proto=%s)", *proto)
	}
	runServer(fmt.Sprintf("%s:%d", *host, *port))
}

type GetWeatherParams struct {
	City string `json:"city" jsonschema:"City name to get weather for"`
}

type WeatherResponse struct {
	CurrentCondition []CurrentCondition `json:"current_condition"`
	NearestArea      []NearestArea      `json:"nearest_area"`
	Weather          []Weather          `json:"weather"`
}

type CurrentCondition struct {
	TempC          string `json:"temp_C"`
	TempF          string `json:"temp_F"`
	FeelsLikeC     string `json:"FeelsLikeC"`
	FeelsLikeF     string `json:"FeelsLikeF"`
	Humidity       string `json:"humidity"`
	Pressure       string `json:"pressure"`
	UvIndex        string `json:"uvIndex"`
	Visibility     string `json:"visibility"`
	WindspeedKmph  string `json:"windspeedKmph"`
	Winddir16Point string `json:"winddir16Point"`
	WeatherDesc    []struct {
		Value string `json:"value"`
	} `json:"weatherDesc"`
	ObservationTime string `json:"observation_time"`
}

type NearestArea struct {
	AreaName []struct {
		Value string `json:"value"`
	} `json:"areaName"`
	Country []struct {
		Value string `json:"value"`
	} `json:"country"`
}

type Weather struct {
	Date     string `json:"date"`
	MaxtempC string `json:"maxtempC"`
	MintempC string `json:"mintempC"`
	Hourly   []struct {
		Humidity      string `json:"humidity"`
		WindspeedKmph string `json:"windspeedKmph"`
		WeatherDesc   []struct {
			Value string `json:"value"`
		} `json:"weatherDesc"`
	} `json:"hourly"`
}

func getWeather(ctx context.Context, req *mcp.CallToolRequest, params *GetWeatherParams) (*mcp.CallToolResult, any, error) {

	client := &http.Client{
		Timeout: 10 * time.Second,
	}

	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	cityName := strings.Replace(params.City, " ", "+", -1)

	request, err := http.NewRequestWithContext(ctx, "GET", "https://wttr.in/+"+cityName+"?format=j1", nil)
	if err != nil {
		return &mcp.CallToolResult{
			Content: []mcp.Content{
				&mcp.TextContent{Text: ""},
			},
		}, nil, fmt.Errorf("failed to create request: %w", err)

	}

	resp, err := client.Do(request)
	if err != nil {
		return &mcp.CallToolResult{
			Content: []mcp.Content{
				&mcp.TextContent{Text: ""},
			},
		}, nil, fmt.Errorf("failed to make request: %w", err)

	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return &mcp.CallToolResult{
			Content: []mcp.Content{
				&mcp.TextContent{Text: ""},
			},
		}, nil, fmt.Errorf("API returned status %d", resp.StatusCode)
	}

	var weatherData WeatherResponse
	if err := json.NewDecoder(resp.Body).Decode(&weatherData); err != nil {
		return &mcp.CallToolResult{
			Content: []mcp.Content{
				&mcp.TextContent{Text: ""},
			},
		}, nil, fmt.Errorf("failed to decode JSON: %w", err)
	}

	if len(weatherData.CurrentCondition) == 0 || len(weatherData.NearestArea) == 0 {
		return &mcp.CallToolResult{
			Content: []mcp.Content{
				&mcp.TextContent{Text: ""},
			},
		}, nil, fmt.Errorf("invalid weather data structure")
	}

	current := weatherData.CurrentCondition[0]
	location := weatherData.NearestArea[0]

	weatherInfo := fmt.Sprintf(`🌤️ **Current Weather in Guatemala City**

**Location**: %s, %s

**Current Conditions**:
• Temperature: %s°C (%s°F)
• Feels like: %s°C (%s°F)
• Weather: %s
• Humidity: %s%%
• Wind: %s km/h %s
• Pressure: %s mb
• UV Index: %s
• Visibility: %s km

**Last Updated**: %s`,
		location.AreaName[0].Value,
		location.Country[0].Value,
		current.TempC,
		current.TempF,
		current.FeelsLikeC,
		current.FeelsLikeF,
		current.WeatherDesc[0].Value,
		current.Humidity,
		current.WindspeedKmph,
		current.Winddir16Point,
		current.Pressure,
		current.UvIndex,
		current.Visibility,
		current.ObservationTime,
	)
	return &mcp.CallToolResult{
		Content: []mcp.Content{
			&mcp.TextContent{Text: weatherInfo},
		},
	}, nil, nil
}

func runServer(url string) {
	server := mcp.NewServer(&mcp.Implementation{
		Name:    "weather-mcp",
		Version: "1.0.0",
	}, nil)

	mcp.AddTool(server, &mcp.Tool{
		Name:        "city_weather",
		Description: "Get the weather statistics for given city name",
	}, getWeather)

	handler := mcp.NewStreamableHTTPHandler(func(req *http.Request) *mcp.Server {
		return server
	}, nil)

	handlerWithLogging := loggingHandler(handler)

	log.Printf("MCP server listening on %s", url)
	log.Printf("Available tool: cityWeather")

	// Start the HTTP server with logging handler.
	if err := http.ListenAndServe(url, handlerWithLogging); err != nil {
		log.Fatalf("Server failed: %v", err)
	}
}
