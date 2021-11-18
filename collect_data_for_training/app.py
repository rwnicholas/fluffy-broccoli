from flask import Flask, Response
from apscheduler.schedulers.background import BackgroundScheduler
import collect

app = Flask(__name__)

@app.route('/')
def update():
    print("Mining Economiza Alagoas")
    collect.run()
    print("Scheduling Economiza Alagoas")
    scheduler = BackgroundScheduler()
    scheduler.add_job(collect.run, 'interval', days=3)
    scheduler.start()

    return Response(status=200)

app.run(host='0.0.0.0', port=7394, use_reloader=False)