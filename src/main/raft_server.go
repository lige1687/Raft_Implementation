package main

import (
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"time"

	raftpkg "./raft"
)

func main() {
	var (
		id      = flag.Int("id", 0, "this node id")
		port    = flag.Int("port", 8080, "http port")
		peers   = flag.String("peers", "", "comma separated peer addresses (not used in this demo)")
		lr      = flag.Float64("lr", 0.001, "learning rate")
	)
	flag.Parse()

	log.Printf("启动Raft节点 id=%d http_port=%d lr=%.4f peers=%s", *id, *port, *lr, *peers)

	// 这里只演示HTTP路径，真正的Raft集群初始化沿用项目原有初始化流程
	// 复用一个最小化的Raft实例以便状态查询/Leader判断
	applyCh := make(chan raftpkg.ApplyMsg, 100)
	persister := raftpkg.MakePersister()
	// peers在本简化版不使用，真实落地请按原项目传入
	rf := raftpkg.Make([]*raftpkg.ClientEnd{}, *id, persister, applyCh)

	service := raftpkg.NewHTTPService(rf, float32(*lr))
	mux := http.NewServeMux()
	service.RegisterHTTPRoutes(mux)

	srv := &http.Server{
		Addr:              fmt.Sprintf(":%d", *port),
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}

	go func() {
		log.Printf("HTTP服务启动于 :%d", *port)
		if err := srv.ListenAndServe(); err != nil {
			log.Printf("HTTP服务退出: %v", err)
		}
	}()

	// 简单阻塞
	select {}
}
