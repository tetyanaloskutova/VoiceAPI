from django.contrib.auth.models import User
from django.contrib.auth import get_backends
from django.contrib.auth import login

class AutoAuthMiddleware():
    """
        Middleware for testing purpose only.
        Can enforce the user login.
    """

    def process_request(self, request):

        user = User.objects.filter()[0]
        if user:
            backend = get_backends()[0]
            user.backend = "%s.%s" % (backend.__module__, backend.__class__.__name__) #fake authentication
            login(request, user)