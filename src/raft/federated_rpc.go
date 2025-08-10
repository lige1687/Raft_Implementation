package raft

import (
	"time"
)

// SubmitModelUpdateArgs 提交模型更新的参数
type SubmitModelUpdateArgs struct {
	Update    ModelUpdate `json:"update"`
	Term      int         `json:"term"`      // 当前任期
	LeaderID  int         `json:"leader_id"` // Leader节点ID
	Timestamp time.Time   `json:"timestamp"`
}

// SubmitModelUpdateReply 提交模型更新的回复
type SubmitModelUpdateReply struct {
	Success   bool   `json:"success"`
	Message   string `json:"message"`
	Term      int    `json:"term"`      // 当前任期
	LeaderID  int    `json:"leader_id"` // 当前Leader
}

// RequestAggregationArgs 请求模型聚合的参数
type RequestAggregationArgs struct {
	RoundID   int       `json:"round_id"`
	ClientID  string    `json:"client_id"`
	Term      int       `json:"term"`
	LeaderID  int       `json:"leader_id"`
	Timestamp time.Time `json:"timestamp"`
}

// RequestAggregationReply 请求模型聚合的回复
type RequestAggregationReply struct {
	Success     bool         `json:"success"`
	GlobalModel *GlobalModel `json:"global_model"`
	Message     string       `json:"message"`
	Term        int          `json:"term"`
	LeaderID    int          `json:"leader_id"`
}

// GetGlobalModelArgs 获取全局模型的参数
type GetGlobalModelArgs struct {
	ClientID  string    `json:"client_id"`
	RoundID   int       `json:"round_id"`
	Term      int       `json:"term"`
	Timestamp time.Time `json:"timestamp"`
}

// GetGlobalModelReply 获取全局模型的回复
type GetGlobalModelReply struct {
	Success     bool         `json:"success"`
	GlobalModel *GlobalModel `json:"global_model"`
	Message     string       `json:"message"`
	Term        int          `json:"term"`
	LeaderID    int          `json:"leader_id"`
}

// SubmitModelUpdate 处理客户端模型更新提交
func (rf *Raft) SubmitModelUpdate(args *SubmitModelUpdateArgs, reply *SubmitModelUpdateReply) {
	rf.mu.Lock()
	defer rf.mu.Unlock()
	
	// 检查是否为Leader
	if rf.state != Leader {
		reply.Success = false
		reply.Message = "Not leader"
		reply.Term = rf.currentTerm
		reply.LeaderID = rf.votedFor // 可能的Leader
		return
	}
	
	// 检查任期
	if args.Term < rf.currentTerm {
		reply.Success = false
		reply.Message = "Stale term"
		reply.Term = rf.currentTerm
		reply.LeaderID = rf.me
		return
	}
	
	// 处理模型更新
	err := rf.federatedManager.SubmitModelUpdate(&args.Update)
	if err != nil {
		reply.Success = false
		reply.Message = err.Error()
		reply.Term = rf.currentTerm
		reply.LeaderID = rf.me
		return
	}
	
	// 将联邦学习操作作为Raft日志条目
	federatedLogEntry := FederatedLogEntry{
		Type:        "model_update",
		RoundID:     args.Update.RoundID,
		ClientID:    args.Update.ClientID,
		ModelUpdate: &args.Update,
		Timestamp:   time.Now(),
	}
	
	// 提交到Raft日志
	logEntry := LogEntry{
		Term:    rf.currentTerm,
		Command: federatedLogEntry,
	}
	
	rf.logs = append(rf.logs, logEntry)
	
	// 触发复制（这里简化处理，实际需要等待多数派确认）
	go rf.replicateToFollowers(logEntry)
	
	reply.Success = true
	reply.Message = "Model update submitted successfully"
	reply.Term = rf.currentTerm
	reply.LeaderID = rf.me
}

// RequestAggregation 处理聚合请求
func (rf *Raft) RequestAggregation(args *RequestAggregationArgs, reply *RequestAggregationReply) {
	rf.mu.Lock()
	defer rf.mu.Unlock()
	
	// 检查是否为Leader
	if rf.state != Leader {
		reply.Success = false
		reply.Message = "Not leader"
		reply.Term = rf.currentTerm
		reply.LeaderID = rf.votedFor
		return
	}
	
	// 检查任期
	if args.Term < rf.currentTerm {
		reply.Success = false
		reply.Message = "Stale term"
		reply.Term = rf.currentTerm
		reply.LeaderID = rf.me
		return
	}
	
	// 触发聚合
	globalModel, err := rf.federatedManager.TriggerAggregation(args.RoundID)
	if err != nil {
		reply.Success = false
		reply.Message = err.Error()
		reply.Term = rf.currentTerm
		reply.LeaderID = rf.me
		return
	}
	
	// 将聚合结果作为Raft日志条目
	federatedLogEntry := FederatedLogEntry{
		Type:      "aggregation",
		RoundID:   args.RoundID,
		ClientID:  args.ClientID,
		Timestamp: time.Now(),
	}
	
	logEntry := LogEntry{
		Term:    rf.currentTerm,
		Command: federatedLogEntry,
	}
	
	rf.logs = append(rf.logs, logEntry)
	
	// 触发复制
	go rf.replicateToFollowers(logEntry)
	
	reply.Success = true
	reply.GlobalModel = globalModel
	reply.Message = "Aggregation completed successfully"
	reply.Term = rf.currentTerm
	reply.LeaderID = rf.me
}

// GetGlobalModel 获取全局模型
func (rf *Raft) GetGlobalModel(args *GetGlobalModelArgs, reply *GetGlobalModelReply) {
	rf.mu.RLock()
	defer rf.mu.RUnlock()
	
	// 检查任期
	if args.Term < rf.currentTerm {
		reply.Success = false
		reply.Message = "Stale term"
		reply.Term = rf.currentTerm
		reply.LeaderID = rf.me
		return
	}
	
	// 获取全局模型
	globalModel := rf.federatedManager.GetGlobalModel()
	
	reply.Success = true
	reply.GlobalModel = globalModel
	reply.Message = "Global model retrieved successfully"
	reply.Term = rf.currentTerm
	
	// 标识当前Leader
	if rf.state == Leader {
		reply.LeaderID = rf.me
	} else {
		reply.LeaderID = rf.votedFor
	}
}

// replicateToFollowers 复制日志到Follower（简化实现）
func (rf *Raft) replicateToFollowers(entry LogEntry) {
	// 这里是简化实现，实际应该使用现有的Raft复制机制
	// 在实际实现中，应该通过现有的AppendEntries RPC来处理
	
	for i := range rf.peers {
		if i != rf.me {
			go func(peerIndex int) {
				// 发送AppendEntries RPC到Follower
				args := &AppendEntriesArgs{
					Term:         rf.currentTerm,
					LeaderId:     rf.me,
					PrevLogIndex: len(rf.logs) - 2, // 前一个日志索引
					PrevLogTerm:  0,
					Entries:      []LogEntry{entry},
					LeaderCommit: rf.commitIndex,
				}
				
				reply := &AppendEntriesReply{}
				rf.peers[peerIndex].Call("Raft.AppendEntries", args, reply)
				
				// 处理回复（简化）
				if reply.Success {
					DPrintf("Successfully replicated federated entry to peer %d", peerIndex)
				} else {
					DPrintf("Failed to replicate federated entry to peer %d", peerIndex)
				}
			}(i)
		}
	}
}

// applyFederatedCommand 应用联邦学习命令到状态机
func (rf *Raft) applyFederatedCommand(cmd interface{}) {
	if federatedEntry, ok := cmd.(FederatedLogEntry); ok {
		switch federatedEntry.Type {
		case "model_update":
			// 应用模型更新
			if federatedEntry.ModelUpdate != nil {
				err := rf.federatedManager.SubmitModelUpdate(federatedEntry.ModelUpdate)
				if err != nil {
					DPrintf("Failed to apply model update: %v", err)
				} else {
					DPrintf("Successfully applied model update from client %s", 
						federatedEntry.ClientID)
				}
			}
			
		case "aggregation":
			// 应用聚合操作
			_, err := rf.federatedManager.TriggerAggregation(federatedEntry.RoundID)
			if err != nil {
				DPrintf("Failed to apply aggregation: %v", err)
			} else {
				DPrintf("Successfully applied aggregation for round %d", 
					federatedEntry.RoundID)
			}
			
		case "validation":
			// 应用验证操作
			DPrintf("Applied validation operation for round %d", federatedEntry.RoundID)
			
		default:
			DPrintf("Unknown federated operation type: %s", federatedEntry.Type)
		}
	}
}


