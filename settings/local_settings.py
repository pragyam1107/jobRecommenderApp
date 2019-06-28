DEBUG = True
LOG_FILENAME = "flask-scaffolding.log"

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s %(levelname)s %(name)s %(message)s'
        }
    },
    'handlers': {
        'default': {
            'level': 'DEBUG',
            'class': 'logging.FileHandler',
            'formatter': 'default',
            'encoding': 'utf-8',
            'filename': LOG_FILENAME
        }
    },
    'loggers': {
        'default': {
            'handlers': ['default'],
            'level': 'DEBUG',
            'propogate': True,
        }
    }
}


# MONGO_SETTINGS = {
#     'DB_NAME': 'scaffolding',
#     'DB_HOST': 'localhost',
#     'DB_PORT': 27017,
#     'DB_USERNAME': '',
#     'DB_PASSWORD': '',
# }
