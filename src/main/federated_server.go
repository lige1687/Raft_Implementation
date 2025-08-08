package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"time"

	"6.5840/raft"
)

// FederatedServer HTTP服务器，为Python客户端提供RESTful API
type FederatedServer struct {
	port     int
	raftNode *raft.Raft
}

// NewFederatedServer 创建联邦学习HTTP服务器
func NewFederatedServer(port int, raftNode *raft.Raft) *FederatedServer {
	return &FederatedServer{
		port:     port,
		raftNode: raftNode,
	}
}

// HTTPResponse 标准HTTP响应格式
type HTTPResponse struct {
	Success   bool        `json:"success"`
	Message   string      `json:"message"`
	Data      interface{} `json:"data,omitempty"`
	Timestamp int64       `json:"timestamp"`
}

// respondJSON 发送JSON响应
func (fs *FederatedServer) respondJSON(w http.ResponseWriter, status int, response HTTPResponse) {
	response.Timestamp = time.Now().Unix()
	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(response)
}

// handleOptions 处理CORS预检请求
func (fs *FederatedServer) handleOptions(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type")
	w.WriteHeader(http.StatusOK)
}

// handleSubmitModelUpdate 处理模型更新提交
func (fs *FederatedServer) handleSubmitModelUpdate(w http.ResponseWriter, r *http.Request) {
	if r.Method == "OPTIONS" {
		fs.handleOptions(w, r)
		return
	}
	
	if r.Method != "POST" {
		fs.respondJSON(w, http.StatusMethodNotAllowed, HTTPResponse{
			Success: false,
			Message: "Method not allowed",
		})
		return
	}
	
	// 解析请求
	var req raft.SubmitModelUpdateArgs
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fs.respondJSON(w, http.StatusBadRequest, HTTPResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to parse request: %v", err),
		})
		return
	}
	
	// 调用Raft RPC
	var reply raft.SubmitModelUpdateReply
	fs.raftNode.SubmitModelUpdate(&req, &reply)
	
	// 构造响应
	response := HTTPResponse{
		Success: reply.Success,
		Message: reply.Message,
		Data: map[string]interface{}{
			"term":      reply.Term,
			"leader_id": reply.LeaderID,
		},
	}
	
	status := http.StatusOK
	if !reply.Success {
		status = http.StatusBadRequest
		if reply.Message == "Not leader" {
			status = http.StatusServiceUnavailable
		}
	}
	
	fs.respondJSON(w, status, response)
	
	log.Printf("Handled model update submission from client %s: success=%v", 
		req.Update.ClientID, reply.Success)
}

// handleRequestAggregation 处理聚合请求
func (fs *FederatedServer) handleRequestAggregation(w http.ResponseWriter, r *http.Request) {
	if r.Method == "OPTIONS" {
		fs.handleOptions(w, r)
		return
	}
	
	if r.Method != "POST" {
		fs.respondJSON(w, http.StatusMethodNotAllowed, HTTPResponse{
			Success: false,
			Message: "Method not allowed",
		})
		return
	}
	
	// 解析请求
	var req raft.RequestAggregationArgs
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fs.respondJSON(w, http.StatusBadRequest, HTTPResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to parse request: %v", err),
		})
		return
	}
	
	// 调用Raft RPC
	var reply raft.RequestAggregationReply
	fs.raftNode.RequestAggregation(&req, &reply)
	
	// 构造响应
	response := HTTPResponse{
		Success: reply.Success,
		Message: reply.Message,
		Data: map[string]interface{}{
			"global_model": reply.GlobalModel,
			"term":         reply.Term,
			"leader_id":    reply.LeaderID,
		},
	}
	
	status := http.StatusOK
	if !reply.Success {
		status = http.StatusBadRequest
		if reply.Message == "Not leader" {
			status = http.StatusServiceUnavailable
		}
	}
	
	fs.respondJSON(w, status, response)
	
	log.Printf("Handled aggregation request from client %s for round %d: success=%v", 
		req.ClientID, req.RoundID, reply.Success)
}

// handleGetGlobalModel 处理获取全局模型请求
func (fs *FederatedServer) handleGetGlobalModel(w http.ResponseWriter, r *http.Request) {
	if r.Method == "OPTIONS" {
		fs.handleOptions(w, r)
		return
	}
	
	if r.Method != "POST" {
		fs.respondJSON(w, http.StatusMethodNotAllowed, HTTPResponse{
			Success: false,
			Message: "Method not allowed",
		})
		return
	}
	
	// 解析请求
	var req raft.GetGlobalModelArgs
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		fs.respondJSON(w, http.StatusBadRequest, HTTPResponse{
			Success: false,
			Message: fmt.Sprintf("Failed to parse request: %v", err),
		})
		return
	}
	
	// 调用Raft RPC
	var reply raft.GetGlobalModelReply
	fs.raftNode.GetGlobalModel(&req, &reply)
	
	// 构造响应
	response := HTTPResponse{
		Success: reply.Success,
		Message: reply.Message,
		Data: map[string]interface{}{
			"global_model": reply.GlobalModel,
			"term":         reply.Term,
			"leader_id":    reply.LeaderID,
		},
	}
	
	status := http.StatusOK
	if !reply.Success {
		status = http.StatusBadRequest
	}
	
	fs.respondJSON(w, status, response)
	
	log.Printf("Handled get global model request from client %s: success=%v", 
		req.ClientID, reply.Success)
}

// handleRaftStatus 处理Raft状态查询
func (fs *FederatedServer) handleRaftStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method == "OPTIONS" {
		fs.handleOptions(w, r)
		return
	}
	
	if r.Method != "GET" {
		fs.respondJSON(w, http.StatusMethodNotAllowed, HTTPResponse{
			Success: false,
			Message: "Method not allowed",
		})
		return
	}
	
	// 获取Raft状态
	term, isLeader := fs.raftNode.GetState()
	
	response := HTTPResponse{
		Success: true,
		Message: "Raft status retrieved",
		Data: map[string]interface{}{
			"term":      term,
			"is_leader": isLeader,
			"node_id":   fs.raftNode.GetNodeID(), // 需要在Raft中添加这个方法
		},
	}
	
	fs.respondJSON(w, http.StatusOK, response)
}

// handleHealth 健康检查
func (fs *FederatedServer) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method == "OPTIONS" {
		fs.handleOptions(w, r)
		return
	}
	
	response := HTTPResponse{
		Success: true,
		Message: "Server is healthy",
		Data: map[string]interface{}{
			"port":   fs.port,
			"uptime": time.Now().Unix(),
		},
	}
	
	fs.respondJSON(w, http.StatusOK, response)
}

// handleStats 获取联邦学习统计信息
func (fs *FederatedServer) handleStats(w http.ResponseWriter, r *http.Request) {
	if r.Method == "OPTIONS" {
		fs.handleOptions(w, r)
		return
	}
	
	if r.Method != "GET" {
		fs.respondJSON(w, http.StatusMethodNotAllowed, HTTPResponse{
			Success: false,
			Message: "Method not allowed",
		})
		return
	}
	
	// 获取联邦学习统计信息
	// 这里需要从Raft节点获取统计信息
	stats := map[string]interface{}{
		"total_rounds":     0,
		"active_clients":   0,
		"total_updates":    0,
		"last_round_time":  time.Now().Unix(),
	}
	
	response := HTTPResponse{
		Success: true,
		Message: "Statistics retrieved",
		Data:    stats,
	}
	
	fs.respondJSON(w, http.StatusOK, response)
}

// setupRoutes 设置HTTP路由
func (fs *FederatedServer) setupRoutes() *http.ServeMux {
	mux := http.NewServeMux()
	
	// 联邦学习API
	mux.HandleFunc("/federated/submit_update", fs.handleSubmitModelUpdate)
	mux.HandleFunc("/federated/request_aggregation", fs.handleRequestAggregation)
	mux.HandleFunc("/federated/get_global_model", fs.handleGetGlobalModel)
	
	// Raft状态API
	mux.HandleFunc("/raft/status", fs.handleRaftStatus)
	
	// 监控API
	mux.HandleFunc("/health", fs.handleHealth)
	mux.HandleFunc("/stats", fs.handleStats)
	
	// 根路径
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		
		fs.respondJSON(w, http.StatusOK, HTTPResponse{
			Success: true,
			Message: "TrustedFL Federated Learning Server",
			Data: map[string]interface{}{
				"version": "1.0.0",
				"endpoints": []string{
					"/federated/submit_update",
					"/federated/request_aggregation", 
					"/federated/get_global_model",
					"/raft/status",
					"/health",
					"/stats",
				},
			},
		})
	})
	
	return mux
}

// Start 启动HTTP服务器
func (fs *FederatedServer) Start() {
	mux := fs.setupRoutes()
	
	addr := fmt.Sprintf(":%d", fs.port)
	log.Printf("Starting Federated Learning HTTP server on port %d", fs.port)
	log.Printf("Available endpoints:")
	log.Printf("  POST /federated/submit_update")
	log.Printf("  POST /federated/request_aggregation")
	log.Printf("  POST /federated/get_global_model")
	log.Printf("  GET  /raft/status")
	log.Printf("  GET  /health")
	log.Printf("  GET  /stats")
	
	server := &http.Server{
		Addr:         addr,
		Handler:      mux,
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}
	
	if err := server.ListenAndServe(); err != nil {
		log.Fatalf("Failed to start HTTP server: %v", err)
	}
}

// 用于演示的主函数（实际使用时应该集成到现有的Raft启动代码中）
func main() {
	if len(os.Args) < 2 {
		log.Fatal("Usage: go run federated_server.go <node_id>")
	}
	
	nodeID, err := strconv.Atoi(os.Args[1])
	if err != nil {
		log.Fatalf("Invalid node ID: %v", err)
	}
	
	// 这里应该是实际的Raft初始化代码
	// 为了演示，我们创建一个模拟的Raft节点
	log.Printf("Starting Federated Learning server for Raft node %d", nodeID)
	
	// 启动HTTP服务器（在实际环境中，这应该在Raft初始化后启动）
	httpPort := 8080 + nodeID
	
	// 注意：这里需要实际的Raft实例
	// server := NewFederatedServer(httpPort, raftNode)
	// server.Start()
	
	log.Printf("Federated Learning server would start on port %d", httpPort)
	log.Printf("To integrate with existing Raft implementation:")
	log.Printf("1. Add this server to your Raft node initialization")
	log.Printf("2. Start the HTTP server after Raft is ready")
	log.Printf("3. Ensure the federated manager is initialized")
}
