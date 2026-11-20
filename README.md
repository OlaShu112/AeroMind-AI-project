
# AeroMind AI – Intelligent Drone Simulation and RAG Assistant

AeroMind AI is an AI-powered application that combines conversational artificial intelligence, Retrieval-Augmented Generation (RAG), real-time information retrieval and drone simulation.

The system provides a web-based Streamlit interface powered by Ollama and Llama 3.2. It integrates Brave Search, OpenAlex and Open-Meteo for external information retrieval.

AeroMind also incorporates PX4 Software-in-the-Loop (SITL), Gazebo and MAVSDK to simulate and control an X500 quadcopter in a virtual environment.

The existing flight-control system supports supervised takeoff, hovering, telemetry monitoring and landing. Natural-language drone control through the chatbot is a planned enhancement.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Backend development and application logic |
| Streamlit | Web-based AI chatbot GUI |
| Ollama | Local AI model runtime |
| Llama 3.2 | Conversational AI |
| HybridRAG | Retrieval-Augmented Generation |
| Brave Search API | Web search |
| OpenAlex API | Academic research retrieval |
| Open-Meteo API | Weather information |
| MAVSDK Python | Drone communication and flight control |
| MAVLink | Communication protocol |
| PX4 Autopilot | Drone flight-control software |
| PX4 SITL | Software-in-the-Loop simulation |
| Gazebo Sim | 3D drone simulation environment |
| Ubuntu WSL2 | Linux development environment |
| Windows 11 | Host operating system |
| Visual Studio Code | Development environment |
| Git | Version control |

## Key Features

- AI-powered chatbot using Ollama and Llama 3.2.
- Local information retrieval using HybridRAG.
- Web search through Brave Search.
- Academic research retrieval through OpenAlex.
- Weather information through Open-Meteo.
- Interactive Streamlit web interface.
- PX4 SITL and Gazebo integration.
- X500 quadcopter simulation.
- MAVSDK-based drone communication.
- Drone connection and flight-readiness checks.
- Battery and position telemetry monitoring.
- Supervised takeoff to 3 metres.
- Ten-second hover sequence.
- Controlled landing.
- Modular architecture for future AI-based drone control.

## System Requirements

### Operating System

- Windows 11.
- Windows Subsystem for Linux (WSL2).
- Ubuntu Linux.

### Required Software

- Python 3.
- pip.
- Python virtual environment.
- Ollama.
- Llama 3.2.
- Streamlit.
- MAVSDK Python.
- PX4 Autopilot.
- Gazebo Sim.
- Visual Studio Code.

## Project Structure

The main AeroMind components are organised as follows:

    aeromind-ai/
    │
    ├── .venv/
    │
    ├── ai/
    │   └── hybrid_rag.py
    │
    ├── gui/
    │   └── rag_chat.py
    │
    ├── flight_control/
    │   └── drone_controller.py
    │
    ├── simulation/
    │   └── first_flight.py
    │
    └── README.md

### Windows Project Location

    C:\xampp\htdocs\aeromind-ai

### Ubuntu Project Location

    /mnt/c/xampp/htdocs/aeromind-ai

### PX4 Installation Location

    /home/olaswe/PX4-Autopilot

## System Architecture

AeroMind AI consists of two main subsystems: the AI assistant and the PX4 drone simulation environment.

    AeroMind AI
         |
         v
    Streamlit GUI
         |
         +-----------------------+
         |                       |
         v                       v
    AI Assistant            Flight Control
         |                       |
         v                       v
    Ollama / Llama 3.2      DroneController
         |                       |
         v                       v
    HybridRAG                 MAVSDK
         |                       |
         v                       v
    Knowledge Retrieval       MAVLink
                                 |
                                 v
                              PX4 SITL
                                 |
                                 v
                               Gazebo
                                 |
                                 v
                              X500 Drone

The AI assistant and drone simulation can currently be operated independently. Direct chatbot-based flight commands require additional integration.

## Installation and Setup

### Step 1 – Open Ubuntu Using WSL

Open Windows PowerShell or Windows Terminal.

Run:

    wsl

Alternatively, open Ubuntu directly from the Windows Start menu.

To list installed Linux distributions:

    wsl --list --verbose

To launch Ubuntu explicitly:

    wsl -d Ubuntu

Use the actual distribution name if it differs.

### Step 2 – Navigate to the AeroMind Project

From Ubuntu:

    cd /mnt/c/xampp/htdocs/aeromind-ai

Verify the current directory:

    pwd

List project files:

    ls -la

### Step 3 – Activate the Python Virtual Environment

Run:

    source .venv/bin/activate

The terminal should display a prompt similar to:

    (.venv) olaswe@Daughty:/mnt/c/xampp/htdocs/aeromind-ai$

Verify Python:

    python --version

Verify pip:

    python -m pip --version

### Step 4 – Install Python Dependencies

If the project contains a requirements file:

    python -m pip install -r requirements.txt

Otherwise, install missing core packages:

    python -m pip install streamlit ollama mavsdk requests

Additional dependencies may be required by the existing HybridRAG implementation.

### Step 5 – Configure Ollama

Verify the Ollama installation:

    ollama --version

List installed models:

    ollama list

Install Llama 3.2 if necessary:

    ollama pull llama3.2

Start the Ollama service if it is not already running:

    ollama serve

Keep the terminal open.

To test the model in another terminal:

    ollama run llama3.2

Example question:

    Explain how PX4 SITL works with Gazebo.

Exit the interactive model:

    /bye

To verify the Ollama API:

    curl http://localhost:11434/api/tags


## Database Architecture and Configuration

AeroMind AI uses PostgreSQL as its proposed primary database for persistent application data.

PostgreSQL stores structured information, including users, chatbot conversations, messages, simulation sessions, flight commands, telemetry records and retrieved research metadata.

The pgvector extension supports vector similarity searches for Retrieval-Augmented Generation (RAG).

### Database Technology

| Component | Technology |
|---|---|
| Relational Database | PostgreSQL |
| Vector Database Extension | pgvector |
| Python ORM | SQLAlchemy |
| Database Driver | psycopg |
| Schema Migrations | Alembic |
| Environment Configuration | python-dotenv |

### Database Architecture

    AeroMind AI
         |
         v
    Streamlit GUI
         |
         v
    Python Application
         |
         +-----------------------+
         |                       |
         v                       v
    AI / HybridRAG          Flight Controller
         |                       |
         +-----------+-----------+
                     |
                     v
              SQLAlchemy ORM
                     |
                     v
                PostgreSQL
                     |
          +----------+----------+
          |                     |
          v                     v
    Application Tables       pgvector
          |                     |
          v                     v
    Users, Messages,       Document Embeddings
    Flights, Telemetry     and Similarity Search

### Step 1 – Install PostgreSQL in Ubuntu

Open an Ubuntu terminal.

Update the package index:

    sudo apt update

Install PostgreSQL:

    sudo apt install postgresql postgresql-contrib -y

Verify PostgreSQL:

    psql --version

### Step 2 – Start PostgreSQL

Start the PostgreSQL service:

    sudo service postgresql start

Check its status:

    sudo service postgresql status

PostgreSQL should report that the service is running.

### Step 3 – Create the AeroMind Database

Open the PostgreSQL administrator shell:

    sudo -u postgres psql

Create the application user:

    CREATE USER aeromind_user WITH PASSWORD 'REPLACE_WITH_STRONG_PASSWORD';

Create the database:

    CREATE DATABASE aeromind_db OWNER aeromind_user;

Exit PostgreSQL:

    \q

Replace the example password with a strong, unique password.

Do not store real database passwords in source code or README files.

### Step 4 – Install pgvector

Install a pgvector package compatible with the installed PostgreSQL server version.

Check the PostgreSQL version:

    pg_config --version

On Ubuntu, the package may be named:

    postgresql-<version>-pgvector

For example, if PostgreSQL 16 is installed and the package is available:

    sudo apt install postgresql-16-pgvector

Replace the version number with the installed PostgreSQL major version.

If the package is unavailable, use the official pgvector installation instructions for the installed PostgreSQL version.

After installation, connect to the AeroMind database:

    sudo -u postgres psql -d aeromind_db

Enable the extension:

    CREATE EXTENSION IF NOT EXISTS vector;

Verify:

    SELECT extname FROM pg_extension WHERE extname = 'vector';

Exit:

    \q

### Step 5 – Install Python Database Dependencies

Navigate to the AeroMind project:

    cd /mnt/c/xampp/htdocs/aeromind-ai

Activate the virtual environment:

    source .venv/bin/activate

Install the required packages:

    python -m pip install sqlalchemy "psycopg[binary]" python-dotenv pgvector alembic

### Step 6 – Configure Environment Variables

Create a `.env` file in the AeroMind project root:

    nano .env

Add:

    DB_HOST=127.0.0.1
    DB_PORT=5432
    DB_NAME=aeromind_db
    DB_USER=aeromind_user
    DB_PASSWORD=REPLACE_WITH_STRONG_PASSWORD

Save the file.

The database password must match the password configured for the PostgreSQL user.

Add the following entries to `.gitignore`:

    .env
    .venv/
    __pycache__/
    *.pyc

Never commit database credentials to GitHub.

### Step 7 – Create the Database Directory

From the project root:

    mkdir -p database

Create the Python package:

    touch database/__init__.py

Create the database connection module:

    touch database/connection.py

### Step 8 – Configure the Python Database Connection

Open:

    database/connection.py

Add the following Python code:

    import os
    from pathlib import Path

    from dotenv import load_dotenv
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import URL
    from sqlalchemy.orm import sessionmaker

    PROJECT_ROOT = Path(__file__).resolve().parent.parent

    load_dotenv(PROJECT_ROOT / ".env")

    DATABASE_URL = URL.create(
        drivername="postgresql+psycopg",
        username=os.environ["DB_USER"],
        password=os.environ["DB_PASSWORD"],
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.environ["DB_NAME"],
    )

    engine = create_engine(
        DATABASE_URL,
        pool_pre_ping=True,
    )

    SessionLocal = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
    )

    def test_connection():
        with engine.connect() as connection:
            result = connection.execute(
                text("SELECT current_database()")
            )

            return result.scalar_one()

This module establishes the reusable database engine and session factory.

### Step 9 – Test the Database Connection

Navigate to the project root:

    cd /mnt/c/xampp/htdocs/aeromind-ai

Activate the environment:

    source .venv/bin/activate

Run:

    python -c "from database.connection import test_connection; print(test_connection())"

Expected output:

    aeromind_db

This confirms that Python can connect to PostgreSQL.

### Step 10 – Configure Database Models and Migrations

The database schema should be defined using SQLAlchemy models.

Create:

    database/models.py

Initial application tables should include:

- users
- chat_sessions
- chat_messages
- knowledge_documents
- document_chunks
- simulation_sessions
- flight_commands
- flight_telemetry
- research_records

Use Alembic to manage schema migrations rather than manually modifying production tables.

Initialise Alembic:

    alembic init database/migrations

Configure Alembic to use the application's database connection and SQLAlchemy model metadata.

After defining the models and configuring Alembic, generate the initial migration:

    alembic revision --autogenerate -m "Initial AeroMind database schema"

Review the generated migration before applying it.

Apply the migration:

    alembic upgrade head

The migration commands require valid SQLAlchemy models and a configured Alembic environment.

### Step 11 – Connect the Database to Streamlit

The Streamlit application can import the shared database session factory:

    from database.connection import SessionLocal

Example:

    with SessionLocal() as db:
        # Execute application database operations here.
        pass

The Streamlit application must be updated to save and retrieve records using this connection.

Creating the database connection alone does not automatically enable chat history persistence.

### Step 12 – Connect PostgreSQL to HybridRAG

HybridRAG can use PostgreSQL and pgvector to store and retrieve document embeddings.

The proposed workflow is:

    Document Upload
          |
          v
    Document Processing
          |
          v
    Text Chunking
          |
          v
    Embedding Model
          |
          v
    PostgreSQL + pgvector
          |
          v
    Similarity Search
          |
          v
    Retrieved Context
          |
          v
    Ollama / Llama 3.2
          |
          v
    Generated Answer

An embedding model must be configured separately from the language model.

Document embeddings must use a consistent vector dimension and compatible embedding model for both indexing and querying.

The existing HybridRAG implementation must be updated to use this database-backed retrieval pipeline.

### Step 13 – Connect Drone Simulation Records

The flight-control subsystem can store:

- Simulation session identifiers.
- Connection status.
- Flight commands.
- Takeoff and landing timestamps.
- Relative altitude.
- Battery telemetry.
- Vehicle status.
- Flight errors.

Telemetry should be written asynchronously or through a separate persistence component so database delays do not block flight-control operations.

Database availability must not be required for emergency landing or PX4 safety functions.

### Step 14 – Verify the Database

Connect using the application credentials:

    psql -h 127.0.0.1 -U aeromind_user -d aeromind_db

List tables:

    \dt

List extensions:

    \dx

Exit:

    \q

Tables will appear after the application schema migrations have been applied successfully.

### Step 15 – Starting AeroMind with PostgreSQL

Start PostgreSQL:

    sudo service postgresql start

Start Ollama if necessary:

    ollama serve

Start the AeroMind GUI in a separate terminal:

    cd /mnt/c/xampp/htdocs/aeromind-ai
    source .venv/bin/activate
    python -m streamlit run gui/rag_chat.py

Open:

    http://localhost:8501

Start PX4 and Gazebo in another terminal:

    cd ~/PX4-Autopilot
    make px4_sitl gz_x500

Start the supervised first-flight script in another terminal:

    cd /mnt/c/xampp/htdocs/aeromind-ai
    source .venv/bin/activate
    python -m simulation.first_flight

The database connection and persistence integrations must be implemented before AeroMind can save chatbot and flight records automatically.

## Running the AeroMind AI GUI

AeroMind uses Streamlit to provide its interactive chatbot interface.

### Step 1 – Open Ubuntu

From Windows PowerShell:

    wsl

### Step 2 – Navigate to AeroMind

    cd /mnt/c/xampp/htdocs/aeromind-ai

### Step 3 – Activate the Virtual Environment

    source .venv/bin/activate

### Step 4 – Start the Streamlit Application

    python -m streamlit run gui/rag_chat.py

### Step 5 – Open the GUI

Open a Windows browser and navigate to:

    http://localhost:8501

The AeroMind chatbot should appear if the application starts successfully.

### Step 6 – Interact with AeroMind

Example questions:

    What is PX4 Autopilot?

    Explain the architecture of a drone.

    What is Retrieval-Augmented Generation?

    Find research papers about autonomous drone navigation.

    What is the weather forecast for Sheffield?

The chatbot processes supported requests through the configured AI and retrieval services.

## Retrieval-Augmented Generation (RAG)

RAG is an artificial intelligence technique that combines information retrieval with language-model generation.

Instead of relying entirely on the language model's training data, a RAG system retrieves relevant information from available knowledge sources and supplies it as context.

### How AeroMind RAG Works

    User Question
         |
         v
    Streamlit Chatbot
         |
         v
    Question Classification
         |
         v
    HybridRAG
         |
         v
    Local Knowledge Retrieval
         |
         v
    Retrieved Context
         |
         v
    Ollama / Llama 3.2
         |
         v
    Generated Answer
         |
         v
    Streamlit GUI

### Local RAG

Local RAG is intended to answer questions using documents and information accessible through AeroMind's knowledge retrieval implementation.

Example:

    Explain the drone architecture in my project documentation.

The relevant documents must be available to the configured retrieval system.

### General AI Questions

AeroMind can use Llama 3.2 for general questions.

Example:

    Explain the difference between AI and machine learning.

### Web Search

AeroMind includes Brave Search integration for retrieving external information.

Example:

    Search the internet for recent PX4 autopilot developments.

### Academic Research

AeroMind includes OpenAlex integration for discovering academic research.

Example:

    Find academic papers about autonomous drone navigation.

### Weather Information

AeroMind includes Open-Meteo integration for weather information.

Example:

    What is the weather forecast for Sheffield?

External services require internet connectivity and appropriate configuration.

## PX4 SITL and Gazebo Simulation

PX4 SITL allows PX4 autopilot software to run in a simulated environment without requiring a physical flight controller.

Gazebo provides the virtual environment, drone model, physics and simulated sensors.

### Simulation Architecture

    Python Flight Script
            |
            v
          MAVSDK
            |
            v
          MAVLink
            |
            v
         PX4 SITL
            |
            v
          Gazebo
            |
            v
      Virtual X500 Drone

## Starting PX4 and Gazebo

### Step 1 – Open Ubuntu Terminal One

Navigate to the PX4 directory:

    cd ~/PX4-Autopilot

### Step 2 – Start the X500 Simulation

    make px4_sitl gz_x500

The command builds required components when necessary and launches the PX4 simulation.

### Step 3 – Verify the Simulation

Successful startup may display:

    INFO [init] Gazebo world is ready
    INFO [init] Spawning Gazebo model
    INFO [gz_bridge] world: default, model: x500_0
    INFO [px4] Startup script returned successfully
    pxh>

The `pxh>` prompt is the PX4 shell.

Keep this terminal running throughout the simulation.

### Step 4 – Open the Gazebo GUI

The PX4 launch command normally attempts to start the Gazebo graphical interface.

If the GUI does not appear, open another Ubuntu terminal.

Check the display environment:

    echo $DISPLAY

Check Wayland:

    echo $WAYLAND_DISPLAY

Attempt to open the graphical client:

    gz sim -g

Do not start another complete simulation world while PX4 is already running.

## Using the Gazebo GUI

The Gazebo GUI provides a three-dimensional representation of the simulated X500 drone.

Users can:

- View the virtual drone.
- Inspect the environment.
- Rotate the camera.
- Zoom in and out.
- Observe takeoff.
- Monitor hovering.
- Observe landing.
- Inspect simulation behaviour.

The GUI displays the simulation, while PX4 performs flight-control operations.

## MAVSDK and Drone Communication

MAVSDK provides the communication interface between Python and PX4.

The previously observed PX4 configuration uses MAVLink UDP port `14540`.

### MAVSDK Connection Endpoint

    udpin://0.0.0.0:14540

The endpoint must match the configuration in the existing DroneController.

### Verify the Connection Configuration

From the AeroMind project directory:

    grep -nE '14540|system_address|connect' flight_control/drone_controller.py

### Check MAVLink Status

At the PX4 shell:

    mavlink status

## Flying the Drone in Simulation

AeroMind includes an existing flight script:

    simulation/first_flight.py

The script uses the existing DroneController from:

    flight_control/drone_controller.py

### Flight Configuration

| Parameter | Value |
|---|---|
| Drone Model | PX4 X500 |
| Target Altitude | 3 metres |
| Hover Duration | 10 seconds |
| Altitude Tolerance | 0.6 metres |
| Climb Timeout | 30 seconds |
| Flight Environment | PX4 SITL and Gazebo |
| Communication | MAVSDK / MAVLink |

### Step 1 – Keep PX4 and Gazebo Running

In Terminal One:

    cd ~/PX4-Autopilot
    make px4_sitl gz_x500

Leave the terminal open.

### Step 2 – Open Ubuntu Terminal Two

Open another Ubuntu terminal.

Navigate to AeroMind:

    cd /mnt/c/xampp/htdocs/aeromind-ai

### Step 3 – Activate the Virtual Environment

    source .venv/bin/activate

### Step 4 – Run the Flight Script

    python -m simulation.first_flight

### Step 5 – Wait for Flight Readiness

The script attempts to:

1. Connect to PX4.
2. Verify position readiness.
3. Check battery telemetry and charge.
4. Request operator confirmation.

If these checks succeed, it displays:

    Gazebo X500 ready? Type TAKEOFF to fly:

### Step 6 – Confirm Takeoff

Type:

    TAKEOFF

Press Enter.

### Step 7 – Takeoff

The controller requests takeoff to 3 metres.

The script monitors relative altitude during the climb.

### Step 8 – Hover

After reaching the required altitude, the script performs a 10-second hover sequence.

### Step 9 – Land

The script checks whether the drone is airborne and attempts a controlled landing when necessary.

If landing cannot be confirmed, it reports an error.

### Step 10 – Verify the Flight

Observe the Gazebo GUI and confirm that the drone:

- Takes off.
- Reaches approximately the target altitude.
- Hovers.
- Lands.

Review the terminal output for telemetry and errors.

## Running AeroMind and Gazebo Simultaneously

Use four Ubuntu terminals.

### Terminal One – PX4 and Gazebo

    cd ~/PX4-Autopilot
    make px4_sitl gz_x500

### Terminal Two – Ollama

    ollama serve

Only start Ollama if it is not already running.

### Terminal Three – AeroMind GUI

    cd /mnt/c/xampp/htdocs/aeromind-ai
    source .venv/bin/activate
    python -m streamlit run gui/rag_chat.py

Open:

    http://localhost:8501

### Terminal Four – Drone Flight

    cd /mnt/c/xampp/htdocs/aeromind-ai
    source .venv/bin/activate
    python -m simulation.first_flight

When prompted:

    TAKEOFF

Only confirm takeoff when the simulation and readiness checks are functioning correctly.

## PX4 Diagnostic Commands

The PX4 shell provides commands for inspecting vehicle status and telemetry.

| Command | Description |
|---|---|
| `commander check` | Check vehicle readiness |
| `commander status` | Inspect commander state if supported |
| `listener vehicle_status` | Display vehicle status |
| `listener vehicle_local_position` | Display local position |
| `listener sensor_accel` | Display accelerometer data |
| `listener sensor_gyro` | Display gyroscope data |
| `mavlink status` | Inspect MAVLink communication |
| `help` | Display available PX4 commands |

### Check Flight Readiness

    commander check

### Check Vehicle Status

    listener vehicle_status

### Check Position

    listener vehicle_local_position

### Check Accelerometer

    listener sensor_accel

### Check MAVLink

    mavlink status

Enter these commands at the `pxh>` prompt, not the normal Ubuntu shell.

## Natural-Language Drone Control

AeroMind is designed to support future natural-language flight commands through the chatbot.

Examples include:

    Open Gazebo and start the X500 simulation.

    Check whether the drone is ready to fly.

    Take off to 3 metres.

    Hover for 10 seconds.

    Land the drone.

These commands are planned capabilities. They require additional integration between the chatbot and DroneController.

### Proposed Command Architecture

    User Command
         |
         v
    AeroMind GUI
         |
         v
    Command Classification
         |
         v
    Command Validation
         |
         v
    Operator Confirmation
         |
         v
    DroneController
         |
         v
    MAVSDK
         |
         v
    PX4 SITL
         |
         v
    Gazebo X500
         |
         v
    Telemetry Feedback

### Flight Safety Requirements

- Restrict flight commands to the simulator.
- Validate all supported commands.
- Require operator confirmation for takeoff.
- Verify vehicle readiness.
- Monitor telemetry.
- Preserve PX4 preflight checks.
- Reject unsupported commands.
- Prevent arbitrary shell-command execution.
- Provide error reporting and controlled landing procedures.

## Troubleshooting

### Error: No Module Named `simulation.first_slight`

The correct filename is:

    first_flight.py

Run:

    python -m simulation.first_flight

### Error: `simulation_controller.py` Not Found

The project already has an existing controller and flight script.

Run:

    python -m simulation.first_flight

### Error: PX4 Server Already Running

Example:

    INFO [px4] PX4 server already running for instance 0

Stop the existing simulation normally.

Inspect processes:

    pgrep -af 'px4|gz'

If an unwanted PX4 process remains:

    pkill -x px4

Restart:

    cd ~/PX4-Autopilot
    make px4_sitl gz_x500

### Gazebo GUI Does Not Open

Check the display environment:

    echo $DISPLAY
    echo $WAYLAND_DISPLAY

Attempt to open the graphical client:

    gz sim -g

### MAVSDK Connection Failure

Check PX4 MAVLink status:

    mavlink status

Inspect the controller configuration:

    grep -nE '14540|system_address|connect' flight_control/drone_controller.py

### Drone Does Not Take Off

Possible causes include:

- PX4 preflight failure.
- Accelerometer timeout.
- Gyroscope or IMU errors.
- Missing position estimates.
- Gazebo simulation being paused.
- Battery telemetry problems.
- MAVSDK communication issues.

At the PX4 shell:

    commander check

Do not bypass PX4 preflight checks to force takeoff.

### Streamlit Not Found

Activate the environment:

    source .venv/bin/activate

Install Streamlit if required:

    python -m pip install streamlit

Start the GUI:

    python -m streamlit run gui/rag_chat.py

### Ollama Connection Refused

Check the Ollama API:

    curl http://localhost:11434/api/tags

Start Ollama if necessary:

    ollama serve

## Stopping the Application

### Step 1 – Complete the Flight

Allow the flight script to finish and confirm the drone has landed.

### Step 2 – Stop Streamlit

Press `Ctrl+C` in the Streamlit terminal.

### Step 3 – Stop PX4

Use the normal PX4 shutdown procedure or stop the foreground simulation.

### Step 4 – Close Gazebo

Close the Gazebo graphical interface after the simulation stops.

### Step 5 – Stop Ollama

If started manually, press `Ctrl+C` in its terminal.

### Step 6 – Exit Ubuntu

    exit

### Step 7 – Shut Down WSL

From Windows PowerShell:

    wsl --shutdown

This stops all running WSL distributions and their processes.

## Quick Start Commands

### 1. Open Ubuntu

    wsl

### 2. Start PX4 and Gazebo

    cd ~/PX4-Autopilot
    make px4_sitl gz_x500

### 3. Start Ollama

    ollama serve

Only if Ollama is not already running.

### 4. Start AeroMind GUI

    cd /mnt/c/xampp/htdocs/aeromind-ai
    source .venv/bin/activate
    python -m streamlit run gui/rag_chat.py

### 5. Open the Chatbot

    http://localhost:8501

### 6. Fly the Simulated Drone

    cd /mnt/c/xampp/htdocs/aeromind-ai
    source .venv/bin/activate
    python -m simulation.first_flight

When prompted:

    TAKEOFF

## Future Development

Planned improvements include:

- Natural-language drone commands.
- Automated Gazebo simulation launching.
- Real-time telemetry dashboards.
- AI-assisted mission planning.
- Waypoint navigation.
- Autonomous navigation experiments.
- Flight logging and analytics.
- Enhanced RAG knowledge management.
- Multi-drone simulation support.

These are planned enhancements rather than confirmed implemented capabilities.

## Conclusion

AeroMind AI integrates conversational artificial intelligence, Retrieval-Augmented Generation, external information retrieval and drone simulation into a modular Python application.

Streamlit provides the chatbot interface, Ollama and Llama 3.2 provide local AI capabilities, and HybridRAG supports knowledge retrieval.

PX4 SITL and Gazebo provide the simulated flight environment, while MAVSDK and the existing DroneController support communication with the X500 quadcopter.

The existing first-flight script provides a supervised workflow for takeoff to 3 metres, hovering for 10 seconds and controlled landing.

The architecture establishes a foundation for future natural-language flight control, real-time telemetry monitoring and AI-assisted autonomous simulation.

**AeroMind AI – Intelligent Assistance, Knowledge Retrieval and Drone Simulation.**
