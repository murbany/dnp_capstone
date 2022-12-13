from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from functools import wraps
import json
import os

from flask import abort, Blueprint, render_template, request, Response
from flask import current_app as app

import load_model

# global model, tokenizer, generator
global generator

global survey_form

with open('./questionnaires/dataset_survey.json') as f:
    survey_form = json.load(f)
    survey_form = survey_form["questions"]
    temp_survey = {}
    for i, q in enumerate(survey_form):
        temp_survey[i] = {
            'question': q["label"],
            'helper': q.get("help", "")
        }
    survey_form = temp_survey

# model, tokenizer, generator = load_model.init()
generator = load_model.init()

QUESTIONNAIRE_DEFAULTS = {
    "submit": "Submit",
    "messages": {
        "error": {
            "required": "Field is required",
            "invalid": "Invalid value"
        },
        "success": "Thank you! Your form has been submitted!"
    }
}

SUBMISSION_DATEFMT = '%Y%m%d%H%M%S%f'


survey = Blueprint('survey', __name__, template_folder='templates')

def _merge_objects(obj1, obj2):
    """Recursive merge obj2 into obj1. Objects can be dicts and lists."""
    if type(obj1) == list:
        obj1.extend(obj2)
        return obj1
    for k2, v2 in obj2.items():
        v1 = obj1.get(k2)
        if type(v2) == type(v1) in (dict, list):
            _merge_objects(v1, v2)
        else:
            obj1[k2] = v2
    return obj1


def _get_option(opt, val=None):
    opt = 'QUESTIONNAIRE_' + opt
    try:
        val = app.config[opt]
    except KeyError:
        if val is None:
            abort(500, "%s is not configured" % opt)
    return val


def _get_defaults():
    return _merge_objects(deepcopy(QUESTIONNAIRE_DEFAULTS),
                          _get_option('DEFAULTS', {}))


def _post_process_data(data):
    for q in data.get('questions', []):
        # find and index "other" options
        if 'options' in q:
            for i, o in enumerate(q['options']):
                if o.startswith('+'):
                    if 'other_option' in q:
                        abort(500, "only one \"other\" option per question is "
                              "allowed: %s" % o)
                    q['options'][i] = q['other_option'] = o[1:]
    return data


def __get_questionnaire_data(slug):
    """Read questionnaire data from the file pointed by slug."""
    qfile = os.path.join(_get_option('DIR'), slug + '.json')
    try:
        with open(qfile) as f:
            data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError('json top level structure must be object')
    except (TypeError, ValueError, OverflowError) as e:
        app.logger.exception('parse error: %s: %s', qfile, e)
        abort(500, "error in %s" % slug)
    except EnvironmentError as e:
        app.logger.info('I/O error: %s: %s', qfile, e)
        abort(404, "Questionnaire not found: %s" % slug)
    if 'extends' in data:
        data = _merge_objects(__get_questionnaire_data(data['extends']), data)
    else:
        data = _merge_objects(_get_defaults(), data)
    return data


def _get_questionnaire_data(slug):
    return _post_process_data(__get_questionnaire_data(slug))


def _get_submissions(slug):
    sdir = os.path.join(_get_option('SUBMISSIONS_DIR'), slug)
    submissions = {}
    try:
        dirlist = os.listdir(sdir)
    except EnvironmentError as e:
        if e.errno != os.errno.ENOENT:
            raise
        dirlist = []
    for subm in filter(lambda s: s.endswith('.json'), dirlist):
            with open(os.path.join(sdir, subm)) as f:
                data = json.load(f)
                data = dict(((int(i), [tuple(v) for v in vs])
                             for i, vs in data.iteritems()))
                dt = datetime.strptime(subm[:-5], SUBMISSION_DATEFMT)
                submissions[dt] = data
    return submissions


def _write_submission(data, slug):
    sdir = os.path.join(_get_option('SUBMISSIONS_DIR'), slug)
    try:
        os.makedirs(sdir)
    except OSError:
        pass
    timestamp = datetime.now().strftime(SUBMISSION_DATEFMT)
    sfile = os.path.join(sdir, timestamp + '.json')
    print(sfile)
    with open(sfile, 'w') as f:
        json.dump(data, f, indent=4)
    print('written')


# HTTP Basic Auth
# http://flask.pocoo.org/snippets/8/

def check_auth(username, password):
    """This function is called to check if a username /
    password combination is valid.
    """
    return _get_option('BASIC_AUTH') == (username, password)


def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})


def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated


@survey.route('/')
@requires_auth
def index():
    dir_ = _get_option('DIR')
    qs = map(lambda s: (s[:-5], _get_questionnaire_data(s[:-5])),
             filter(lambda s: not s.startswith('_') and s.endswith('.json'),
                    os.listdir(dir_)))
    return render_template('index.html', questionnaires=qs)


@survey.route('/<slug>', methods=['GET', 'POST'])
def questionnaire(slug):
    data = _get_questionnaire_data(slug)
    template = data.get('template', 'questionnaire.html')

    if request.method == 'GET':
        return render_template(template, questionnaire=data, slug=slug, templates = os.listdir('./string_templates'))
    else:
        submissions = defaultdict(lambda: tuple(None, None))
        form = request.form
        for name, values in form.lists():
            if name == 'submit' or name.endswith('.other') or name=='predict':
                continue
            valid = name.startswith('q')
            if valid:
                try:
                    n = int(name[1:None])
                except ValueError:
                    valid = False
            if not valid:
                app.logger.info('invalid parameter: %s', name)
                return "invalid submission parameters", 400
            submissions[n] = list(filter(any,
                ((s.strip(), form.get('%s.%s.other' % (name, s.strip())))
                 for s in values)))

        error = False
        errormsgs = data['messages']['error']

        for i, q in enumerate(data['questions']):
            s = submissions.get(i, [])
            required = q.get('required')
            other_opt = q.get('other_option')
            values = list(v[0] for v in s)
            # empty required field error
            empty_required = bool(required) and not any(values)
            # empty "other" field error
            empty_other = (bool(other_opt) and
                           any(True for v, o in s if v == other_opt and not o))
            if empty_required or empty_other:
                q['error'] = errormsgs['required']
            if values:
                q['value'] = values[0]
                q['values'] = values
            if other_opt:
                other_vals = filter(None, (o for v, o in s if v == other_opt))
                if other_vals:
                    q['other_value'] = other_vals[0]
            # validate option values
            invalid = (bool(values) and 'options' in q and
                       any(True for v in values if v not in q['options']))
            if invalid:
                q['error'] = errormsgs['invalid']
            error = error or empty_required or empty_required or invalid
        if error:
            return render_template(template, questionnaire=data, slug=slug)
        else:
            _write_submission(submissions, slug)
            return render_template("success.html",
                                   message=data['messages']['success'])


@survey.route('/<slug>/results', methods=['GET'])
@requires_auth
def results(slug):
    data = _get_questionnaire_data(slug)
    submissions = _get_submissions(slug)
    return render_template("results.html",
                           questionnaire=data,
                           submissions=submissions)

@survey.route('/<slug>/predict', methods=['POST'])
def predict(slug):
    # data = _get_questionnaire_data(slug)
    # submissions = _get_submissions(slug)
    qid = None
    length = None
    prev_q = None
    survey_response = defaultdict(lambda: None)
    form = request.form

    for name, values in form.lists():
        if name == 'submit' or name.endswith('.other') or name=='predict':
            continue

        if name == "id":
            qid = int(values[0][1:None])
            continue

        if name == "length":
            length = int(values[0])
            continue
            
        if name == "template":
            template = values[0].split('\\')[-1]
            with open('./string_templates/' + template) as f:
                template = json.load(f)
            continue

        if name == "previous_questions":
            prev_q = values[0]
            if prev_q == 'All':
                prev_q = len(survey_form.items())
            else:
                prev_q = int(prev_q)
            continue
        
        if name.endswith("_context"):
            n = int(name[1:2])
            contexts = list(filter(any,
                    ((s.strip(), form.get('%s.%s.other' % (name, s.strip())))
                        for s in values)))
            survey_response[n].update({
                "context": contexts[0][0] if contexts else ""
            })
        else:
            n = int(name[1:None])
            responses = list(filter(any,
                    ((s.strip(), form.get('%s.%s.other' % (name, s.strip())))
                        for s in values)))
            survey_response[n] = {
                "response": responses[0][0] if responses else ""
            }
    
    prompt = template["leading"] 

    if qid == 0:
        prompt += template["last_chunk"].format(
            question = survey_form[qid]["question"],
            helper = survey_form[qid]["helper"],
            response = survey_response[qid]["response"],
        )
    elif qid == 1:
        prompt += template["chunk"].format(
            question = survey_form[qid-1]["question"],
            helper = survey_form[qid-1]["helper"],
            response = survey_response[qid-1]["response"],
        )
        prompt += template["last_chunk"].format(
            question = survey_form[qid]["question"],
            helper = survey_form[qid]["helper"],
            response = survey_response[qid]["response"],
        )
    else:
        print('))))))))))))))))))))))))))))')
        print(range(max(qid-prev_q, 0), qid-1))
        for i in range(max(qid-prev_q, 0), qid-1):
            print(i)
            prompt += template["chunk"].format(
                question = survey_form[i]["question"],
                helper = survey_form[i]["helper"],
                response = survey_response[i]["response"],
            )
        prompt += template["last_chunk"].format(
            question = survey_form[qid]["question"],
            helper = survey_form[qid]["helper"],
            response = survey_response[qid]["response"],
        )

    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # generated_ids = model.generate(input_ids, do_sample=True, max_new_tokens=length)
    # context = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0][len(prompt):]
    # print(len(prompt))
    prompt_size = len(prompt)
    if prompt_size > 9000:
        prompt = prompt[prompt_size-8999:]
    context = generator(prompt, do_sample=True, max_new_tokens=length, max_length=None)[0]['generated_text'][len(prompt):]
    
    # print(len(context), len(prompt))
    return context