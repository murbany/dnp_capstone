# Capstone for Harvard's AC297r: Large Language Model Assisted Survey Webapp 

In collaboration with the Data Nutrition Project (DNP), we created a platform for ML assisted surveys to investigate improvemets in response quality for the existing survey used to generate the DNPs Data Nutrition Label (DNL)

## Installation

Using Python 3.9+

Can install most of the env with conda, but need to run the pip install the torch txt file bc it doesn't play nice with the conda env
u:admin
pw:secret at login


Then create a new application (or embed to the existing one). See
`example_app.py` as for example. Create config similar to
`example_config.py`. It must contain `QUESTIONNAIRE_DIR` which must
point to a directory with questionnaire files and
`QUESTIONNAIRE_SUBMISSIONS_DIR` where submissions will be written.
Don't forget to udpate `QUESTIONNAIRE_BASIC_AUTH` tuple. It is used
for authentication to the questionnaire list and results pages.

## Usage

Site root provides a list of existing questionnaires with links to
forms and results.

Look at `questionnaires/texteditor.json` for a questionnaire example.
Each questionnaire is an object with the following keys:

    extends
    title
    comment
    template
    submit
    questions

With `extends` you can specify a different json file as a base, it
should be prefixed with `_` so it will not be considered a stand-alone
questionnaire. With `template` you can specify alternative template.
If not specified `questionnaire.html` is used. `questions` is a list
of objects which represent (surprisingly) questions. Each such object
must be one of the following types: `string` (text), `text`
(textarea), `radio` and `checkbox`. `radio`, `checkbox` questions must
have `options`. To mark option as "other" field (a field with a text
input) prefix it with "+" sign.

You can override templates by creating templates with the same
names in your application's template directory or you can specify
`template` option for desired questionnaires.