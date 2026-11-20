
-- ============================================================
-- AEROMIND AI
-- COMPLETE MYSQL DATABASE SCHEMA
-- Version: 2.0
-- Database: aeromind_ai
-- Tables: 25
-- Engine: InnoDB
-- Character Set: utf8mb4
-- ============================================================

CREATE DATABASE IF NOT EXISTS aeromind_ai
CHARACTER SET utf8mb4
COLLATE utf8mb4_unicode_ci;

USE aeromind_ai;

-- ============================================================
-- 1. USERS
-- ============================================================

CREATE TABLE IF NOT EXISTS users (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    username VARCHAR(100) NOT NULL UNIQUE,
    email VARCHAR(255) UNIQUE,
    password_hash VARCHAR(255),
    full_name VARCHAR(255),
    role VARCHAR(50) DEFAULT 'user',
    account_status VARCHAR(30) DEFAULT 'active',
    last_login_at DATETIME NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ============================================================
-- 2. USER SETTINGS
-- ============================================================

CREATE TABLE IF NOT EXISTS user_settings (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT UNSIGNED NOT NULL UNIQUE,
    preferred_model VARCHAR(100) DEFAULT 'llama3.2',
    default_chat_mode VARCHAR(50) DEFAULT 'general',
    theme VARCHAR(30) DEFAULT 'dark',
    language VARCHAR(20) DEFAULT 'en',
    save_chat_history BOOLEAN DEFAULT TRUE,
    notifications_enabled BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE
) ENGINE=InnoDB;

-- ============================================================
-- 3. AI MODELS
-- ============================================================

CREATE TABLE IF NOT EXISTS ai_models (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    model_name VARCHAR(150) NOT NULL UNIQUE,
    provider VARCHAR(100) DEFAULT 'ollama',
    model_type VARCHAR(50) DEFAULT 'chat',
    model_version VARCHAR(100),
    context_window INT UNSIGNED,
    temperature DECIMAL(4,2) DEFAULT 0.70,
    endpoint_url VARCHAR(500),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ============================================================
-- 4. CHAT SESSIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS chat_sessions (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT UNSIGNED NULL,
    title VARCHAR(255),
    mode VARCHAR(50) DEFAULT 'general',
    model_id BIGINT UNSIGNED NULL,
    session_status VARCHAR(30) DEFAULT 'active',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE SET NULL,

    FOREIGN KEY (model_id)
        REFERENCES ai_models(id)
        ON DELETE SET NULL
) ENGINE=InnoDB;

-- ============================================================
-- 5. CHAT MESSAGES
-- ============================================================

CREATE TABLE IF NOT EXISTS chat_messages (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_id BIGINT UNSIGNED NOT NULL,
    role VARCHAR(30) NOT NULL,
    content LONGTEXT NOT NULL,
    model_name VARCHAR(100),
    grounded BOOLEAN DEFAULT FALSE,
    response_time_seconds DECIMAL(10,3),
    prompt_tokens INT UNSIGNED,
    completion_tokens INT UNSIGNED,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (session_id)
        REFERENCES chat_sessions(id)
        ON DELETE CASCADE,

    INDEX idx_chat_messages_session_time
        (session_id, created_at)
) ENGINE=InnoDB;

-- ============================================================
-- 6. KNOWLEDGE DOCUMENTS
-- ============================================================

CREATE TABLE IF NOT EXISTS knowledge_documents (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT UNSIGNED NULL,
    filename VARCHAR(255),
    file_path TEXT,
    file_type VARCHAR(50),
    file_size_bytes BIGINT UNSIGNED,
    file_hash CHAR(64),
    category VARCHAR(100),
    indexed BOOLEAN DEFAULT FALSE,
    indexing_status VARCHAR(30) DEFAULT 'pending',
    chunk_count INT UNSIGNED DEFAULT 0,
    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE SET NULL,

    INDEX idx_knowledge_category (category),
    INDEX idx_knowledge_status (indexing_status)
) ENGINE=InnoDB;

-- ============================================================
-- 7. DOCUMENT CHUNKS
-- ============================================================

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    document_id BIGINT UNSIGNED NOT NULL,
    chunk_index INT UNSIGNED NOT NULL,
    chunk_text LONGTEXT NOT NULL,
    token_count INT UNSIGNED,
    page_number INT UNSIGNED,
    metadata_json JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (document_id)
        REFERENCES knowledge_documents(id)
        ON DELETE CASCADE,

    UNIQUE KEY uq_document_chunk
        (document_id, chunk_index)
) ENGINE=InnoDB;

-- ============================================================
-- 8. DOCUMENT EMBEDDINGS
-- ============================================================

CREATE TABLE IF NOT EXISTS document_embeddings (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    chunk_id BIGINT UNSIGNED NOT NULL,
    embedding_model VARCHAR(150) NOT NULL,
    embedding_dimension INT UNSIGNED NOT NULL,
    embedding_vector JSON NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (chunk_id)
        REFERENCES document_chunks(id)
        ON DELETE CASCADE,

    UNIQUE KEY uq_chunk_embedding_model
        (chunk_id, embedding_model)
) ENGINE=InnoDB;

-- NOTE:
-- MySQL JSON stores embedding arrays, but this schema does
-- not provide a native vector similarity index.
-- Similarity calculations require application logic or
-- a separate vector-search engine.

-- ============================================================
-- 9. RETRIEVAL SOURCES
-- ============================================================

CREATE TABLE IF NOT EXISTS retrieval_sources (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    message_id BIGINT UNSIGNED NOT NULL,
    document_id BIGINT UNSIGNED NULL,
    chunk_id BIGINT UNSIGNED NULL,
    source_title VARCHAR(500),
    source_url TEXT,
    source_type VARCHAR(100),
    relevance_score DOUBLE,
    retrieved_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (message_id)
        REFERENCES chat_messages(id)
        ON DELETE CASCADE,

    FOREIGN KEY (document_id)
        REFERENCES knowledge_documents(id)
        ON DELETE SET NULL,

    FOREIGN KEY (chunk_id)
        REFERENCES document_chunks(id)
        ON DELETE SET NULL
) ENGINE=InnoDB;

-- ============================================================
-- 10. SEARCH HISTORY
-- ============================================================

CREATE TABLE IF NOT EXISTS search_history (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT UNSIGNED NULL,
    session_id BIGINT UNSIGNED NULL,
    search_query TEXT NOT NULL,
    search_provider VARCHAR(100),
    search_type VARCHAR(50),
    results_count INT UNSIGNED DEFAULT 0,
    response_time_seconds DECIMAL(10,3),
    searched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE SET NULL,

    FOREIGN KEY (session_id)
        REFERENCES chat_sessions(id)
        ON DELETE SET NULL
) ENGINE=InnoDB;

-- ============================================================
-- 11. API INTEGRATIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS api_integrations (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    provider_name VARCHAR(100) NOT NULL UNIQUE,
    api_category VARCHAR(100),
    base_url VARCHAR(500),
    credential_env_key VARCHAR(150),
    is_enabled BOOLEAN DEFAULT TRUE,
    last_checked_at DATETIME NULL,
    last_status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- API credentials must be stored in environment variables
-- or a secrets manager, not in this table.

-- ============================================================
-- 12. API REQUEST LOGS
-- ============================================================

CREATE TABLE IF NOT EXISTS api_request_logs (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    integration_id BIGINT UNSIGNED NOT NULL,
    request_type VARCHAR(100),
    http_method VARCHAR(10),
    response_status INT,
    duration_ms INT UNSIGNED,
    error_message TEXT,
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (integration_id)
        REFERENCES api_integrations(id)
        ON DELETE CASCADE,

    INDEX idx_api_request_time (requested_at)
) ENGINE=InnoDB;

-- ============================================================
-- 13. DRONES
-- ============================================================

CREATE TABLE IF NOT EXISTS drones (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    drone_code VARCHAR(100) NOT NULL UNIQUE,
    drone_name VARCHAR(255),
    drone_model VARCHAR(100) DEFAULT 'X500',
    autopilot_type VARCHAR(100) DEFAULT 'PX4',
    connection_protocol VARCHAR(50) DEFAULT 'MAVLink',
    connection_address VARCHAR(255),
    environment VARCHAR(50) DEFAULT 'simulation',
    status VARCHAR(50) DEFAULT 'offline',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ============================================================
-- 14. MISSIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS missions (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    mission_code VARCHAR(100) NOT NULL UNIQUE,
    mission_name VARCHAR(255),
    user_id BIGINT UNSIGNED NULL,
    drone_id BIGINT UNSIGNED NULL,
    description TEXT,
    environment VARCHAR(100),
    status VARCHAR(50) DEFAULT 'planned',
    target_altitude_m DOUBLE,
    planned_duration_seconds INT UNSIGNED,
    started_at DATETIME NULL,
    completed_at DATETIME NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE SET NULL,

    FOREIGN KEY (drone_id)
        REFERENCES drones(id)
        ON DELETE SET NULL,

    INDEX idx_mission_status (status)
) ENGINE=InnoDB;

-- ============================================================
-- 15. SIMULATION SESSIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS simulation_sessions (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    mission_id BIGINT UNSIGNED NULL,
    drone_id BIGINT UNSIGNED NOT NULL,
    simulator VARCHAR(100) DEFAULT 'Gazebo',
    simulator_version VARCHAR(100),
    px4_version VARCHAR(100),
    world_name VARCHAR(255),
    vehicle_model VARCHAR(100) DEFAULT 'gz_x500',
    connection_endpoint VARCHAR(255),
    session_status VARCHAR(50) DEFAULT 'created',
    started_at DATETIME NULL,
    ended_at DATETIME NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (mission_id)
        REFERENCES missions(id)
        ON DELETE SET NULL,

    FOREIGN KEY (drone_id)
        REFERENCES drones(id)
) ENGINE=InnoDB;

-- ============================================================
-- 16. MISSION WAYPOINTS
-- ============================================================

CREATE TABLE IF NOT EXISTS mission_waypoints (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    mission_id BIGINT UNSIGNED NOT NULL,
    waypoint_order INT UNSIGNED NOT NULL,
    latitude DOUBLE NOT NULL,
    longitude DOUBLE NOT NULL,
    altitude_m DOUBLE NOT NULL,
    hold_seconds INT UNSIGNED DEFAULT 0,
    waypoint_status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (mission_id)
        REFERENCES missions(id)
        ON DELETE CASCADE,

    UNIQUE KEY uq_mission_waypoint
        (mission_id, waypoint_order)
) ENGINE=InnoDB;

-- ============================================================
-- 17. FLIGHT COMMANDS
-- ============================================================

CREATE TABLE IF NOT EXISTS flight_commands (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    mission_id BIGINT UNSIGNED NULL,
    simulation_session_id BIGINT UNSIGNED NULL,
    user_id BIGINT UNSIGNED NULL,
    command_type VARCHAR(100) NOT NULL,
    command_parameters JSON,
    command_source VARCHAR(50) DEFAULT 'python',
    confirmation_required BOOLEAN DEFAULT FALSE,
    confirmed_at DATETIME NULL,
    command_status VARCHAR(50) DEFAULT 'pending',
    error_message TEXT,
    requested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    executed_at DATETIME NULL,
    completed_at DATETIME NULL,

    FOREIGN KEY (mission_id)
        REFERENCES missions(id)
        ON DELETE SET NULL,

    FOREIGN KEY (simulation_session_id)
        REFERENCES simulation_sessions(id)
        ON DELETE SET NULL,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE SET NULL,

    INDEX idx_flight_command_status (command_status)
) ENGINE=InnoDB;

-- ============================================================
-- 18. FLIGHT TELEMETRY
-- ============================================================

CREATE TABLE IF NOT EXISTS flight_telemetry (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    mission_id BIGINT UNSIGNED NOT NULL,
    simulation_session_id BIGINT UNSIGNED NULL,
    altitude_m DOUBLE,
    relative_altitude_m DOUBLE,
    latitude DOUBLE,
    longitude DOUBLE,
    battery_percent DOUBLE,
    velocity_north_m_s DOUBLE,
    velocity_east_m_s DOUBLE,
    velocity_down_m_s DOUBLE,
    heading_degrees DOUBLE,
    flight_mode VARCHAR(100),
    armed BOOLEAN,
    in_air BOOLEAN,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (mission_id)
        REFERENCES missions(id),

    FOREIGN KEY (simulation_session_id)
        REFERENCES simulation_sessions(id)
        ON DELETE SET NULL,

    INDEX idx_telemetry_mission_time
        (mission_id, recorded_at)
) ENGINE=InnoDB;

-- ============================================================
-- 19. FLIGHT LOGS
-- ============================================================

CREATE TABLE IF NOT EXISTS flight_logs (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    mission_id BIGINT UNSIGNED NOT NULL,
    simulation_session_id BIGINT UNSIGNED NULL,
    log_level VARCHAR(30) DEFAULT 'INFO',
    log_category VARCHAR(100),
    message TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (mission_id)
        REFERENCES missions(id)
        ON DELETE CASCADE,

    FOREIGN KEY (simulation_session_id)
        REFERENCES simulation_sessions(id)
        ON DELETE SET NULL,

    INDEX idx_flight_logs_time (created_at)
) ENGINE=InnoDB;

-- ============================================================
-- 20. FLIGHT SAFETY CHECKS
-- ============================================================

CREATE TABLE IF NOT EXISTS flight_safety_checks (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    mission_id BIGINT UNSIGNED NOT NULL,
    simulation_session_id BIGINT UNSIGNED NULL,
    check_name VARCHAR(150) NOT NULL,
    check_status VARCHAR(50) NOT NULL,
    measured_value VARCHAR(255),
    details TEXT,
    checked_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (mission_id)
        REFERENCES missions(id)
        ON DELETE CASCADE,

    FOREIGN KEY (simulation_session_id)
        REFERENCES simulation_sessions(id)
        ON DELETE SET NULL
) ENGINE=InnoDB;

-- ============================================================
-- 21. SYSTEM EVENTS
-- ============================================================

CREATE TABLE IF NOT EXISTS system_events (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT UNSIGNED NULL,
    mission_id BIGINT UNSIGNED NULL,
    event_type VARCHAR(100),
    severity VARCHAR(30),
    description TEXT,
    component VARCHAR(100),
    metadata_json JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE SET NULL,

    FOREIGN KEY (mission_id)
        REFERENCES missions(id)
        ON DELETE SET NULL,

    INDEX idx_system_events_time (created_at)
) ENGINE=InnoDB;

-- ============================================================
-- 22. APPLICATION SETTINGS
-- ============================================================

CREATE TABLE IF NOT EXISTS application_settings (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    setting_key VARCHAR(150) NOT NULL UNIQUE,
    setting_value TEXT,
    value_type VARCHAR(30) DEFAULT 'string',
    description TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB;

-- ============================================================
-- 23. BACKGROUND JOBS
-- ============================================================

CREATE TABLE IF NOT EXISTS background_jobs (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    job_type VARCHAR(100) NOT NULL,
    job_status VARCHAR(50) DEFAULT 'pending',
    payload JSON,
    attempts INT UNSIGNED DEFAULT 0,
    max_attempts INT UNSIGNED DEFAULT 3,
    error_message TEXT,
    scheduled_at DATETIME NULL,
    started_at DATETIME NULL,
    completed_at DATETIME NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_background_job_status
        (job_status, scheduled_at)
) ENGINE=InnoDB;

-- ============================================================
-- 24. NOTIFICATIONS
-- ============================================================

CREATE TABLE IF NOT EXISTS notifications (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT UNSIGNED NOT NULL,
    notification_type VARCHAR(100),
    title VARCHAR(255) NOT NULL,
    message TEXT,
    severity VARCHAR(30) DEFAULT 'info',
    is_read BOOLEAN DEFAULT FALSE,
    read_at DATETIME NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE CASCADE,

    INDEX idx_notification_user_read
        (user_id, is_read)
) ENGINE=InnoDB;

-- ============================================================
-- 25. AUDIT LOGS
-- ============================================================

CREATE TABLE IF NOT EXISTS audit_logs (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    user_id BIGINT UNSIGNED NULL,
    action VARCHAR(150) NOT NULL,
    entity_type VARCHAR(100),
    entity_id BIGINT UNSIGNED,
    ip_address VARCHAR(45),
    user_agent TEXT,
    details JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (user_id)
        REFERENCES users(id)
        ON DELETE SET NULL,

    INDEX idx_audit_action_time
        (action, created_at)
) ENGINE=InnoDB;

-- ============================================================
-- INITIAL CONFIGURATION DATA
-- ============================================================

INSERT INTO ai_models (
    model_name,
    provider,
    model_type,
    is_active
)
VALUES (
    'llama3.2',
    'ollama',
    'chat',
    TRUE
)
ON DUPLICATE KEY UPDATE
    model_name = VALUES(model_name);

INSERT INTO api_integrations (
    provider_name,
    api_category,
    base_url,
    credential_env_key
)
VALUES
(
    'Brave Search',
    'web_search',
    'https://api.search.brave.com',
    'BSA4KgXpvCIEGcxseIWqAxTXgUqWsG3'
),
(
    'OpenAlex',
    'academic_research',
    'https://api.openalex.org',
    NULL
),
(
    'Open-Meteo',
    'weather',
    'https://api.open-meteo.com',
    NULL
)
ON DUPLICATE KEY UPDATE
    provider_name = VALUES(provider_name);

INSERT INTO application_settings (
    setting_key,
    setting_value,
    value_type,
    description
)
VALUES
(
    'default_ai_model',
    'llama3.2',
    'string',
    'Default Ollama language model'
),
(
    'simulation_only',
    'true',
    'boolean',
    'Restrict drone control to simulation'
),
(
    'default_takeoff_altitude_m',
    '3.0',
    'float',
    'Default simulated takeoff altitude'
),
(
    'default_hover_duration_seconds',
    '10',
    'integer',
    'Default simulated hover duration'
)
ON DUPLICATE KEY UPDATE
    setting_key = VALUES(setting_key);

-- ============================================================
-- VERIFY DATABASE
-- ============================================================

SHOW TABLES;

SELECT COUNT(*) AS total_tables
FROM information_schema.tables
WHERE table_schema = 'aeromind_ai'
AND table_type = 'BASE TABLE';

-- Expected total: 25 tables
-- ============================================================
-- END OF AEROMIND AI DATABASE SCHEMA
-- ============================================================
