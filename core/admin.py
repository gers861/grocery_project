from django.contrib import admin

# Register your models here.
# your_app_name/admin.py

from django.contrib import admin
from .models import User, Product, Cart, CartItem, Order, OrderItem, RecommendationLog, BehaviorLog

# Register your models here.
admin.site.register(User)
admin.site.register(Product)
admin.site.register(Cart)
admin.site.register(CartItem)
admin.site.register(Order)
admin.site.register(OrderItem)
admin.site.register(RecommendationLog)
admin.site.register(BehaviorLog)