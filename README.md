<div>
    <h1 align="center"> Project 1 - Networks 🌎</h1>
    <h3 align="center"> 
        A CLI MCP client and a Remote MCP server
    </h3>
</div>

## Local MCP

https://github.com/DanielRasho/GRAPH-CP 

## Client 🛜

### Requirements
- uv >= 0.8.13
- python >= 3.13

### Installation🏃

1. Create a `mcp_config.json` file with the following structure: 

```json
{
    "servers": [
        // For local servers
        {
          "name": "Git",
          "transport": "stdio",
          "command": "uvx",
          "args": [
            "mcp-server-git",
            "--repository", 
            "./"
          ],
          "description": "File system operations server",
          "env": {
          }
        },
        // For remote servers
        {
          "name": "NixPackages",
          "transport": "sse",
          "url": "http://127.0.0.1:8080/mcp"
        }
    ]
}
```

2. Create and `.env` file:
```env
# Claude API Configuration
ANTHROPIC_API_KEY=<YOUR KEY>
```

3. Finally, execute the client!
```bash
uv --directory=./ run client.py
```

## Local server 🛜

Available here : https://github.com/DanielRasho/GRAPH-CP

## Remote Server (Weather MCP) 🌨️

The remote server is an implementation of a simple weather forecast server to get info about the weather in any city of the world https://wttr.in/ API.

### Requirements 
- Go >= 1.21

### Installation 🏃
Once you have Go, navegate to the folder `./remote-mcp` and run:

```bash
go run . -host <HOST> -port <PORT>
```

### Tools

The server expose a single tool `city_weather` which receives a city name an return varios weather stats like temperature, presure, humidity, visibility...

## Working Demo 🎥

A full demostration is available here, **feel free to jump around the chapters**: 

**https://youtu.be/-wRqb2aPWsc**

Chech the report here:

