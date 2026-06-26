from flask import render_template, Response, jsonify
from config import settings
from src.web.stream import gen_frames, current_stats

def register_routes(app):
    @app.route('/')
    def index():
        return render_template('index.html', streams=settings.STREAMS)

    @app.route('/api/stream/<video_id>')
    def stream(video_id):
        return Response(gen_frames(video_id),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/api/stats/<video_id>')
    def stats(video_id):
        if video_id in current_stats:
            return jsonify(current_stats[video_id])
        else:
            return jsonify({
                "car": 0, "motorcycle": 0, "truck": 0, "bus": 0, "license_plate": 0, "fps": 0.0, "status": "inactive"
            })
