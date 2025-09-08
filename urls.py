# grocery_project/urls.py
# grocery_project/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('api/', include('core.urls')),   # <- ensures /api/checkout/<id>/ exists
]
