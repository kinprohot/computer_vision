from src.web.app import create_app
from src.utils.logger import logger

app = create_app()

if __name__ == "__main__":
    logger.info("Starting Traffic Monitoring Dashboard web server on http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
