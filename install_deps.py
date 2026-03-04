"""Install dependencies for the neuro-symbolic hazard detection project."""

import subprocess
import sys

def install_requirements():
    """Install requirements from requirements.txt."""
    print("Installing dependencies from requirements.txt...")

    try:
        # Read requirements
        with open('requirements.txt', 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        print(f"Found {len(requirements)} dependencies to install")

        # Install each requirement
        for requirement in requirements:
            print(f"Installing {requirement}...")
            result = subprocess.run([
                sys.executable, '-m', 'pip', 'install', requirement
            ], capture_output=True, text=True)

            if result.returncode != 0:
                print(f"Warning: Failed to install {requirement}: {result.stderr}")
            else:
                print(f"✅ Successfully installed {requirement}")

    except FileNotFoundError:
        print("Error: requirements.txt not found")
        return False
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        return False

    return True

def create_docker_services():
    """Create Docker services (Neo4j)."""
    print("\nSetting up Docker services...")

    # Check if Docker is available
    try:
        result = subprocess.run(['docker', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("Docker not found. Please install Docker to use Neo4j.")
            return False
    except FileNotFoundError:
        print("Docker not found. Please install Docker to use Neo4j.")
        return False

    try:
        # Start Neo4j container
        print("Starting Neo4j container...")
        result = subprocess.run([
            'docker-compose', 'up', '-d'
        ], capture_output=True, text=True)

        if result.returncode != 0:
            print(f"Warning: Failed to start Neo4j container: {result.stderr}")
            print("You can start it manually later with: docker-compose up -d")
        else:
            print("✅ Neo4j container started successfully")
            print("📊 Neo4j Browser: http://localhost:7474")
            print("🔌 Bolt connection: bolt://localhost:7687")
    except FileNotFoundError:
        print("docker-compose not found. Please install Docker Compose.")
        return False

    return True

def verify_installation():
    """Verify that installation was successful."""
    print("\nVerifying installation...")

    try:
        # Test imports
        print("Testing imports...")

        # Core packages
        import yaml
        import numpy
        import cv2
        import sklearn

        # Optional packages (may not be installed yet)
        try:
            import shapely
            print("✅ Shapely imported successfully")
        except ImportError:
            print("⚠️  Shapely not installed (required for spatial analysis)")

        try:
            import neo4j
            print("✅ Neo4j imported successfully")
        except ImportError:
            print("⚠️  Neo4j not installed (required for knowledge graph)")

        try:
            from ultralytics import YOLO
            print("✅ Ultralytics YOLO imported successfully")
        except ImportError:
            print("⚠️  Ultralytics not installed (required for object detection)")

        print("✅ Core packages imported successfully")

    except ImportError as e:
        print(f"❌ Import verification failed: {e}")
        return False

    return True

def main():
    """Main installation function."""
    print("🚀 Installing Neuro-Symbolic Hazard Detection Project")
    print("=" * 60)

    # Install requirements
    if not install_requirements():
        print("Installation failed")
        return

    # Setup Docker services
    create_docker_services()

    # Verify installation
    if not verify_installation():
        print("Installation verification failed")
        return

    print("=" * 60)
    print("✅ Installation completed successfully!")
    print("\nNext steps:")
    print("1. Test the pipeline: python pipeline.py --help")
    print("2. Run tests: python test_integration.py")
    print("3. Stop Neo4j: docker-compose down")
    print("\nFor training:")
    print("1. Prepare your dataset")
    print("2. Configure training_config.yaml")
    print("3. Use vision/trainer.py")

if __name__ == "__main__":
    main()