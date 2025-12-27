package com.example.hegrpc.config;

import io.grpc.ManagedChannel;
import io.grpc.ManagedChannelBuilder;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import javax.annotation.PreDestroy;
import java.util.concurrent.TimeUnit;

/**
 * Configuration for gRPC channel connection.
 */
@Configuration
public class GrpcConfig {

    @Value("${grpc.server.host:localhost}")
    private String grpcHost;

    @Value("${grpc.server.port:50051}")
    private int grpcPort;

    private ManagedChannel channel;

    /**
     * Creates a managed gRPC channel as a Spring Bean.
     * This channel is shared across all gRPC calls.
     */
    @Bean
    public ManagedChannel managedChannel() {
        channel = ManagedChannelBuilder
                .forAddress(grpcHost, grpcPort)
                .usePlaintext()  // No TLS (use TLS in production!)
                .keepAliveTime(30, TimeUnit.SECONDS)
                .keepAliveTimeout(10, TimeUnit.SECONDS)
                .build();
        
        return channel;
    }

    /**
     * Gracefully shutdown the channel when Spring context closes.
     */
    @PreDestroy
    public void shutdownChannel() {
        if (channel != null && !channel.isShutdown()) {
            try {
                channel.shutdown().awaitTermination(5, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                channel.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
    }
}
