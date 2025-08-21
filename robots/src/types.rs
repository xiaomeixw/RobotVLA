//! Core types for the RobotVLA execution engine

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

/// Represents a function call to be executed on a robot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Unique identifier for this function call
    pub id: String,
    
    /// Name of the function to call
    pub function_name: String,
    
    /// Parameters to pass to the function
    pub parameters: HashMap<String, serde_json::Value>,
    
    /// Target robot ID (optional, uses default if None)
    pub robot_id: Option<String>,
    
    /// Execution priority (higher = more urgent)
    pub priority: i32,
    
    /// Execution timeout in seconds
    pub timeout: f64,
    
    /// Timestamp when the call was created
    pub created_at: DateTime<Utc>,
}

impl FunctionCall {
    /// Create a new function call
    pub fn new(
        function_name: String,
        parameters: HashMap<String, serde_json::Value>,
    ) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            function_name,
            parameters,
            robot_id: None,
            priority: 0,
            timeout: 30.0,
            created_at: Utc::now(),
        }
    }

    /// Set the robot ID
    pub fn with_robot_id(mut self, robot_id: String) -> Self {
        self.robot_id = Some(robot_id);
        self
    }

    /// Set the priority
    pub fn with_priority(mut self, priority: i32) -> Self {
        self.priority = priority;
        self
    }

    /// Set the timeout
    pub fn with_timeout(mut self, timeout: f64) -> Self {
        self.timeout = timeout;
        self
    }
}

/// Status of function execution
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum ExecutionStatus {
    /// Execution completed successfully
    Success,
    /// Execution failed with error
    Failed,
    /// Execution is in progress
    InProgress,
    /// Execution was cancelled
    Cancelled,
    /// Execution timed out
    Timeout,
}

/// Result of executing a function call
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionResult {
    /// ID of the original function call
    pub id: String,
    
    /// Execution status
    pub status: ExecutionStatus,
    
    /// Function return value (if successful)
    pub result: Option<serde_json::Value>,
    
    /// Error message (if failed)
    pub error: Option<String>,
    
    /// Time taken to execute in seconds
    pub execution_time: f64,
    
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

impl ExecutionResult {
    /// Create a successful result
    pub fn success(id: String, result: Option<serde_json::Value>, execution_time: f64) -> Self {
        Self {
            id,
            status: ExecutionStatus::Success,
            result,
            error: None,
            execution_time,
            metadata: HashMap::new(),
        }
    }

    /// Create a failed result
    pub fn failed(id: String, error: String, execution_time: f64) -> Self {
        Self {
            id,
            status: ExecutionStatus::Failed,
            result: None,
            error: Some(error),
            execution_time,
            metadata: HashMap::new(),
        }
    }

    /// Create a timeout result
    pub fn timeout(id: String, execution_time: f64) -> Self {
        Self {
            id,
            status: ExecutionStatus::Timeout,
            result: None,
            error: Some("Execution timed out".to_string()),
            execution_time,
            metadata: HashMap::new(),
        }
    }
}

/// Schema definition for a robot function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSchema {
    /// Function name
    pub name: String,
    
    /// Function description
    pub description: String,
    
    /// Parameter definitions
    pub parameters: HashMap<String, ParameterSchema>,
    
    /// Return value description
    pub return_type: Option<String>,
    
    /// Safety constraints
    pub safety_constraints: Vec<SafetyConstraint>,
    
    /// Execution timeout override
    pub timeout: Option<f64>,
}

/// Parameter schema for function parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterSchema {
    /// Parameter type (e.g., "float", "int", "string", "array")
    pub param_type: String,
    
    /// Parameter description
    pub description: String,
    
    /// Whether parameter is required
    pub required: bool,
    
    /// Default value if not required
    pub default: Option<serde_json::Value>,
    
    /// Minimum value (for numeric types)
    pub min: Option<f64>,
    
    /// Maximum value (for numeric types)
    pub max: Option<f64>,
    
    /// Valid choices (for enum-like parameters)
    pub choices: Option<Vec<serde_json::Value>>,
}

/// Safety constraint for function execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConstraint {
    /// Constraint type
    pub constraint_type: String,
    
    /// Constraint parameters
    pub parameters: HashMap<String, serde_json::Value>,
    
    /// Error message if constraint is violated
    pub error_message: String,
}

/// Robot action types supported by the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionType {
    /// Position control (x, y, z, rx, ry, rz, gripper)
    Position,
    /// Velocity control
    Velocity,
    /// Force control
    Force,
    /// Joint control
    Joint,
    /// Gripper control
    Gripper,
    /// Composite action (multiple primitives)
    Composite,
}

/// Robot action representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobotAction {
    /// Action type
    pub action_type: ActionType,
    
    /// Action values
    pub values: Vec<f64>,
    
    /// Confidence score
    pub confidence: f64,
    
    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Execution statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionStats {
    /// Total number of functions executed
    pub total_executed: u64,
    
    /// Number of successful executions
    pub successful: u64,
    
    /// Number of failed executions
    pub failed: u64,
    
    /// Number of timed out executions
    pub timeouts: u64,
    
    /// Average execution time in seconds
    pub avg_execution_time: f64,
    
    /// Number of currently active executions
    pub active_executions: u64,
    
    /// Queue sizes per robot
    pub queue_sizes: HashMap<String, usize>,
}

impl Default for ExecutionStats {
    fn default() -> Self {
        Self {
            total_executed: 0,
            successful: 0,
            failed: 0,
            timeouts: 0,
            avg_execution_time: 0.0,
            active_executions: 0,
            queue_sizes: HashMap::new(),
        }
    }
} 