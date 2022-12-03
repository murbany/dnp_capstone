from flask import Flask

from survey import survey


app = application = Flask(__name__)
app.config.from_pyfile('config.py')
app.register_blueprint(survey)


if __name__ == '__main__':
    app.run()
