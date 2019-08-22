#! /usr/bin/python
import sys
import logging
logging.basicConfig(stream=sys.stderr)
sys.path.insert(0,"/var/www/FlaskApplications/SampleApp/api/")

# home points to the home.py file
from demo import app as application
application.secret_key = "somesecretsessionkey"