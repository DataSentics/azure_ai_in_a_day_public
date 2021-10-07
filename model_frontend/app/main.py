
from flask import Flask, render_template, flash, request
import json
import os
import requests
import uuid
from wtforms import (Form,
                     TextField,
                     TextAreaField,
                     validators,
                     StringField,
                     SubmitField,
                     PasswordField,
                     BooleanField,
                     FloatField,
                     IntegerField,
                     )

# DEBUG = True
HOST = os.environ["FLASK_HOST"]
PORT = os.environ["FLASK_PORT"]
DEBUG = True if os.environ["FLASK_DEBUG"].lower() == "true" else False
SCORING_ENDPOINT = os.environ["SCORING_ENDPOINT"]

app = Flask(__name__)
app.config.from_object(__name__)
# app.config["SECRET_KEY"] = os.environ["FLASK_SECRET_KEY"]
app.config["SECRET_KEY"] = "verysecretkey1234"


class MyForm(Form):
    pass
    # field = StringField("param", [validators.DataRequired()])

    # username = StringField('Username', [validators.Length(min=4, max=25)])
    # email = StringField('Email Address', [validators.Length(min=6, max=35)])
    # password = PasswordField('New Password', [
    #     validators.DataRequired(),
    #     validators.EqualTo('confirm', message='Passwords must match')
    # ])
    # confirm = PasswordField('Repeat Password')
    # accept_tos = BooleanField('I accept the TOS', [validators.DataRequired()])


with open("./swagger.json", "r") as infile:
    swagger_definition = json.load(infile)
    fields = swagger_definition["definitions"]["ServiceInput"]["properties"]["data"]["items"]["properties"]
    init_values = swagger_definition["definitions"]["ServiceInput"]["example"]["data"][0]

    for field, type_def in fields.items():
        if type_def["type"] == "boolean":
            setattr(MyForm, field, BooleanField(field,  # 
                default=init_values[field]))
        elif type_def["type"] == "integer":
            setattr(MyForm, field, IntegerField(field, [validators.DataRequired()],
                default=init_values[field]))
        elif type_def["type"] == "number":
            setattr(MyForm, field, FloatField(field, [validators.DataRequired()],
                default=init_values[field]))
        elif type_def["type"] == "string":
            setattr(MyForm, field, StringField(field, [validators.DataRequired()],
                default=init_values[field]))
        else:
            raise ValueError("Unexpected field type '{}' encountered.".format(field["type"]))


def get_prediction(form_data):
    scoring_uri = SCORING_ENDPOINT
    # If the service is authenticated, set the key or token
    # key = '<your key or token>'

    # app.logger.info(form_data)

    # Two sets of data to score, so we get two results back
    data = {"data": [form_data]}
    # Convert to JSON string
    # app.logger.info(data)
    input_data = json.dumps(data)
    # app.logger.info(input_data)

    # Set the content type
    headers = {'Content-Type': 'application/json'}
    # If authentication is enabled, set the authorization header
    # headers['Authorization'] = f'Bearer {key}'

    # Make the request and display the response
    response = requests.post(scoring_uri, input_data, headers=headers)
    # print(response.text)
    return json.loads(response.json())

#@app.route('/register', methods=['GET', 'POST'])
@app.route('/', methods=['GET', 'POST'])
def index():
    form = MyForm(request.form)
    if request.method == 'POST':
        # app.logger.info(str(fields))
        # app.logger.info(str(init_values))
        if form.validate():
            # sanitize missing False values of BooleanField
            data = request.form.to_dict()
            for field, type_def in fields.items():
                if type_def["type"] == "boolean":
                    data[field] = data.get(field, False)

            prediction = get_prediction(data)
            # app.logger.info("dir(prediction): '" + str(dir(prediction)) + "'")
            app.logger.info("prediction: '" + str(prediction) + "'")
            app.logger.info("type(prediction): '" + str(type(prediction)) + "'")
            app.logger.info("dir(prediction): '" + str(dir(prediction)) + "'")
            # for _i in dir(prediction):
            #     app.logger.info("'{}': '{}'".format(_i, str(getattr(prediction, _i))))
            flash("Prediction: '{}'".format(str(prediction["result"])))
        else:
            flash("Invalid parameter!")

    return render_template('index.html', form=form)

if __name__ == "__main__":
    app.run(host=HOST,
            port=PORT,
            debug=DEBUG,
            )
