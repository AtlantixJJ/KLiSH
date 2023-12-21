# -*- coding: utf-8 -*-
from __future__ import unicode_literals

import numpy as np
import traceback
from django.shortcuts import render
from django.template import loader, Context
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from io import BytesIO
from PIL import Image
from base64 import b64encode, b64decode
from datetime import datetime

import home.api as api

EDIT_INDEX = "index_edit.html"
model_manager = api.ModelAPI("home/static/config.json")
editor = api.EditAPI(model_manager)
config = model_manager.config["models"]
model_name = list(config.keys())[0]
imsize = config[model_name]["image_size"]


def image2bytes(image):
    """Encode image as bytes."""
    buffered = BytesIO()
    Image.fromarray(image).save(buffered, format="PNG")
    return b64encode(buffered.getvalue()).decode("utf-8")


def response_image_label(image, label):
    """Response."""
    imageString = image2bytes(image)
    segString = image2bytes(label)
    json = (
        '{"ok":"true","img":"data:image/png;base64,%s","label":"data:image/png;base64,%s"}'
        % (imageString, segString)
    )
    return HttpResponse(json)


def response_image(image):
    """Response."""
    imageString = image2bytes(image)
    json = '{"ok":"true","img":"data:image/png;base64,%s"}' % imageString
    return HttpResponse(json)


def save_to_session(session, z, label, imsize):
    """Using session to store variables."""
    session["z"] = z
    session["label"] = label
    session["imsize"] = imsize


def restore_from_session(session):
    """Using session to store variables."""
    z = session["z"]
    label = session["label"]
    imsize = session["imsize"]
    return z, label, imsize


def index(request):
    """Index page"""
    res = render(request, EDIT_INDEX)
    res.set_cookie("last_visit", datetime.now())
    return res


@csrf_exempt
def generate_image_given_stroke(request):
    """Edit an image using stroke."""
    form_data = request.POST
    sess = request.session
    if request.method == "POST" and "image_stroke" in form_data:
        try:
            model = form_data["model"]
            if not editor.has_model(model):
                print(f"!> Model not exist {model}")
                return HttpResponse("{}")

            label_stroke = b64decode(form_data["label_stroke"].split(",")[1])
            z_arr, orig_label_arr, imsize = restore_from_session(sess)

            label_stroke = Image.open(BytesIO(label_stroke))
            label_stroke, label_mask = api.stroke2array(label_stroke)

            res = editor.generate_image_given_stroke(
                model, z_arr, orig_label_arr, label_stroke, label_mask
            )
            image_arr, label_viz, label_arr, z_arr = res
            save_to_session(sess, z_arr, label_arr, image_arr.shape)
            return response_image_label(image_arr, label_viz)
        except Exception:
            print("!> Exception:")
            traceback.print_exc()
            return HttpResponse("{}")
    print(f"!> Invalid request: {str(form_data.keys())}")
    return HttpResponse("{}")


@csrf_exempt
def generate_new_image(request):
    form_data = request.POST
    sess = request.session

    if request.method == "POST" and "model" in form_data:
        try:
            model = form_data["model"]
            if not editor.has_model(model):
                print("=> No model name %s" % model)
                return HttpResponse("{}")

            image, labelviz_arr, label_arr, z_arr = editor.generate_new_image(model)
            save_to_session(sess, z_arr, label_arr, image.shape)
            return response_image_label(image, labelviz_arr)
        except Exception:
            print("!> Exception:")
            traceback.print_exc()
            return HttpResponse("{}")
    return HttpResponse("{}")


@csrf_exempt
def sample_noise(request):
    form_data = request.POST
    sess = request.session
    if request.method == "POST":
        try:
            model = form_data["model"]
            if not editor.has_model(model):
                print(f"!> Model not exist {model}")
                return HttpResponse("{}")

            z_arr, orig_label_arr, imsize = restore_from_session(sess)
            label_stroke = np.zeros(imsize, dtype="uint8")
            label_mask = np.zeros(imsize[:2], dtype="uint8")

            image_arr, label_viz, label_arr, z_arr = editor.generate_image_given_stroke(
                model, None, orig_label_arr, label_stroke, label_mask
            )
            save_to_session(sess, z_arr, label_arr, image_arr.shape)
            return response_image_label(image_arr, label_viz)
        except Exception:
            print("!> Exception:")
            traceback.print_exc()
            return HttpResponse("{}")
    print(f"!> Invalid request: {str(form_data.keys())}")
    return HttpResponse("{}")
