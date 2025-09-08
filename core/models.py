from django.db import models
from django.contrib.auth.models import AbstractUser
from django.conf import settings


class User(AbstractUser):
    address = models.CharField(max_length=255, blank=True)


class Product(models.Model):
    name = models.CharField(max_length=255)
    category = models.CharField(max_length=100)
    price = models.FloatField()
    description = models.TextField()
    image = models.ImageField(upload_to='product_images/', blank=True)

    def __str__(self):
        return self.name


class Cart(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='carts')
    created_at = models.DateTimeField(auto_now_add=True)


class CartItem(models.Model):
    cart = models.ForeignKey(Cart, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.IntegerField()


class Order(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='orders')
    order_date = models.DateTimeField(auto_now_add=True)
    total_amount = models.FloatField()


class OrderItem(models.Model):
    order = models.ForeignKey(Order, on_delete=models.CASCADE, related_name='items')
    product = models.ForeignKey(Product, on_delete=models.CASCADE)
    quantity = models.IntegerField()
    price = models.FloatField()


class RecommendationLog(models.Model):
    SOURCE_CHOICES = [
        ("ml", "ML"),
        ("cooccurrence", "Co-occurrence"),
        ("popular", "Popular"),
        ("cold_start", "Cold Start"),
    ]

    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, blank=True
    )
    order = models.ForeignKey("Order", on_delete=models.SET_NULL, null=True, blank=True)
    recommendations = models.JSONField()
    source = models.CharField(max_length=32, choices=SOURCE_CHOICES, default="cold_start")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"RecoLog user={self.user_id} order={self.order_id} source={self.source}"


class BehaviorLog(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name='behavior_logs')
    action_type = models.CharField(max_length=255)
    timestamp = models.DateTimeField(auto_now_add=True)
    details = models.TextField()
