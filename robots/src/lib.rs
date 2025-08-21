//! RobotVLA Function Execution Engine
//!
//! A high-performance, async robot function execution engine implemented in Rust.
//! This engine provides:
//! - Function registry and dynamic loading
//! - Concurrent execution with priority queues
//! - Safety monitoring and timeout handling
//! - Python bindings for integration

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use anyhow::{Result, anyhow};

pub mod types;
pub mod registry;
pub mod executor;
pub mod adapters;
pub mod safety;
pub mod python;

pub use types::*;
pub use registry::*;
pub use executor::*;
pub use adapters::*;

/// Main robot function execution engine
#[derive(Debug)]
pub struct RobotVLAEngine {
    /// Function registry for all registered robots
    registry: Arc<RwLock<FunctionRegistry>>,
    
    /// Task executor for managing concurrent execution
    executor: Arc<TaskExecutor>,
    
    /// Safety monitor for ensuring safe operation
    safety_monitor: Arc<SafetyMonitor>,
    
    /// Active robot connections
    robots: Arc<RwLock<HashMap<String, Box<dyn RobotAdapter + Send + Sync>>>>,
    
    /// Engine configuration
    config: EngineConfig,
}

/// Configuration for the robot engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    /// Maximum number of concurrent tasks
    pub max_concurrent_tasks: usize,
    
    /// Default task timeout in seconds
    pub default_timeout: u64,
    
    /// Enable safety monitoring
    pub enable_safety_monitoring: bool,
    
    /// Maximum queue size per robot
    pub max_queue_size: usize,
    
    /// Heartbeat interval for robot connections
    pub heartbeat_interval: u64,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_concurrent_tasks: 10,
            default_timeout: 30,
            enable_safety_monitoring: true,
            max_queue_size: 100,
            heartbeat_interval: 5,
        }
    }
}

impl RobotVLAEngine {
    /// Create a new robot VLA engine
    pub fn new(config: EngineConfig) -> Self {
        let registry = Arc::new(RwLock::new(FunctionRegistry::new()));
        let executor = Arc::new(TaskExecutor::new(config.max_concurrent_tasks));
        let safety_monitor = Arc::new(SafetyMonitor::new());
        let robots = Arc::new(RwLock::new(HashMap::new()));

        Self {
            registry,
            executor,
            safety_monitor,
            robots,
            config,
        }
    }

    /// Register a robot adapter
    pub async fn register_robot(&self, robot: Box<dyn RobotAdapter + Send + Sync>) -> Result<()> {
        let robot_id = robot.get_id().to_string();
        
        // Register robot functions
        let functions = robot.get_available_functions().await?;
        {
            let mut registry = self.registry.write().await;
            registry.register_robot(&robot_id, functions)?;
        }

        // Store robot adapter
        {
            let mut robots = self.robots.write().await;
            robots.insert(robot_id.clone(), robot);
        }

        tracing::info!("Registered robot: {}", robot_id);
        Ok(())
    }

    /// Execute a function call
    pub async fn execute_function(&self, call: FunctionCall) -> Result<ExecutionResult> {
        // Validate function exists
        let registry = self.registry.read().await;
        let robot_id = call.robot_id.as_deref().unwrap_or("default");
        
        if !registry.has_function(robot_id, &call.function_name) {
            return Ok(ExecutionResult {
                id: call.id.clone(),
                status: ExecutionStatus::Failed,
                result: None,
                error: Some(format!("Function {} not found for robot {}", call.function_name, robot_id)),
                execution_time: 0.0,
                metadata: HashMap::new(),
            });
        }

        // Check safety constraints
        if self.config.enable_safety_monitoring {
            if let Err(e) = self.safety_monitor.validate_function_call(&call).await {
                return Ok(ExecutionResult {
                    id: call.id.clone(),
                    status: ExecutionStatus::Failed,
                    result: None,
                    error: Some(format!("Safety check failed: {}", e)),
                    execution_time: 0.0,
                    metadata: HashMap::new(),
                });
            }
        }

        // Submit to executor
        self.executor.execute(call).await
    }

    /// Get available functions for a robot
    pub async fn get_available_functions(&self, robot_id: &str) -> Result<Vec<FunctionSchema>> {
        let registry = self.registry.read().await;
        Ok(registry.get_functions(robot_id))
    }

    /// Get execution statistics
    pub async fn get_stats(&self) -> ExecutionStats {
        self.executor.get_stats().await
    }

    /// Shutdown the engine gracefully
    pub async fn shutdown(&self) -> Result<()> {
        tracing::info!("Shutting down RobotVLA engine");
        
        // Stop executor
        self.executor.shutdown().await;
        
        // Disconnect all robots
        let mut robots = self.robots.write().await;
        for (robot_id, robot) in robots.drain() {
            if let Err(e) = robot.disconnect().await {
                tracing::warn!("Error disconnecting robot {}: {}", robot_id, e);
            }
        }

        tracing::info!("RobotVLA engine shutdown complete");
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[tokio::test]
    async fn test_engine_creation() {
        let config = EngineConfig::default();
        let engine = RobotVLAEngine::new(config);
        
        // Test basic functionality
        let stats = engine.get_stats().await;
        assert_eq!(stats.total_executed, 0);
    }
} 