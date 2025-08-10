package raft

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strconv"
	"sync"
	"time"
)

// HTTPService 提供基于HTTP的梯度提交与聚合查询
// 说明：为快速打通端到端链路，本服务直接复用本地 GradientManager，
// 后续可将聚合结果通过 Raft 日志进行共识提交。
type HTTPService struct {
	mu sync.RWMutex

	raft            *Raft
	gradientManager *GradientManager

	// 聚合结果缓存：batchID -> 聚合梯度
	lastAggregated map[int]*AggregatedGradient
}

// NewHTTPService 创建HTTP服务
func NewHTTPService(rf *Raft, learningRate float32) *HTTPService {
	return &HTTPService{
		raft:            rf,
		gradientManager: NewGradientManager(learningRate),
		lastAggregated:  make(map[int]*AggregatedGradient),
	}
}

// SubmitGradientHandler 处理梯度提交（POST /submit_gradient）
// 请求体为 GradientLog JSON
func (s *HTTPService) SubmitGradientHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		w.WriteHeader(http.StatusMethodNotAllowed)
		return
	}

	decoder := json.NewDecoder(r.Body)
	var grad GradientLog
	if err := decoder.Decode(&grad); err != nil {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(fmt.Sprintf("invalid json: %v", err)))
		return
	}

	// 时间戳填充
	if grad.Timestamp.IsZero() {
		grad.Timestamp = time.Now()
	}

	// 简单Leader校验（非强制）
	_, isLeader := s.raft.GetState()
	if !isLeader {
		// 非Leader也可先接收（后续可做重定向），此处仅告知客户端当前非Leader
		w.Header().Set("X-Not-Leader", "true")
	}

	shouldAggregate := s.gradientManager.AddGradient(grad)
	log.Printf("收到梯度: worker=%d batch=%d params=%d", grad.WorkerID, grad.BatchID, len(grad.Gradients))

	if shouldAggregate {
		agg := s.gradientManager.AggregateGradients(grad.BatchID)
		if agg != nil {
			// 应用到模型
			s.gradientManager.ApplyGradientToModel(agg)
			s.mu.Lock()
			s.lastAggregated[grad.BatchID] = agg
			s.mu.Unlock()
			log.Printf("完成聚合: batch=%d workers=%d", grad.BatchID, agg.WorkerCount)
		}
	}

	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(map[string]interface{}{"ok": true})
}

// GetAggregatedHandler 获取聚合结果（GET /aggregated?batch_id=1）
func (s *HTTPService) GetAggregatedHandler(w http.ResponseWriter, r *http.Request) {
	batchIDStr := r.URL.Query().Get("batch_id")
	if batchIDStr == "" {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("missing batch_id"))
		return
	}
	batchID, err := strconv.Atoi(batchIDStr)
	if err != nil {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte("invalid batch_id"))
		return
	}

	s.mu.RLock()
	agg := s.lastAggregated[batchID]
	s.mu.RUnlock()

	w.Header().Set("Content-Type", "application/json")
	if agg == nil {
		_ = json.NewEncoder(w).Encode(map[string]interface{}{"found": false})
		return
	}
	_ = json.NewEncoder(w).Encode(agg)
}

// RegisterHTTPRoutes 注册路由
func (s *HTTPService) RegisterHTTPRoutes(mux *http.ServeMux) {
	mux.HandleFunc("/submit_gradient", s.SubmitGradientHandler)
	mux.HandleFunc("/aggregated", s.GetAggregatedHandler)
}
