import os
from django.conf import settings
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from rest_framework import status
from .ml_models.loader import recommend_for_user, load_resources # This line is critical




from django.shortcuts import render

# Create your views here.
from rest_framework import viewsets
from .models import User, Product, Cart, CartItem, Order, OrderItem, RecommendationLog, BehaviorLog
from .serializers import (
    UserSerializer, ProductSerializer, CartSerializer, CartItemSerializer,
    OrderSerializer, OrderItemSerializer, RecommendationLogSerializer, BehaviorLogSerializer
)


# --- Load ML resources on server startup ---


class UserViewSet(viewsets.ModelViewSet):
    queryset = User.objects.all()
    serializer_class = UserSerializer

class ProductViewSet(viewsets.ModelViewSet):
    queryset = Product.objects.all()
    serializer_class = ProductSerializer

class CartViewSet(viewsets.ModelViewSet):
    queryset = Cart.objects.all()
    serializer_class = CartSerializer

class CartItemViewSet(viewsets.ModelViewSet):
    queryset = CartItem.objects.all()
    serializer_class = CartItemSerializer

class OrderViewSet(viewsets.ModelViewSet):
    queryset = Order.objects.all()
    serializer_class = OrderSerializer

class OrderItemViewSet(viewsets.ModelViewSet):
    queryset = OrderItem.objects.all()
    serializer_class = OrderItemSerializer

class RecommendationLogViewSet(viewsets.ModelViewSet):
    queryset = RecommendationLog.objects.all()
    serializer_class = RecommendationLogSerializer

class BehaviorLogViewSet(viewsets.ModelViewSet):
    queryset = BehaviorLog.objects.all()
    serializer_class = BehaviorLogSerializer
    
    # core/views.py (add this at the bottom)
@api_view(["POST"])
def ml_recommendations(request):
    """
    POST body:
    {
        "user_id": 1808,
        "basket": ["whole milk", "rolls/buns"],
        "top_k": 5
    }
    """
    try:
        user_id = request.data.get("user_id")
        basket = request.data.get("basket", [])
        top_k = int(request.data.get("top_k", 5))

        if not user_id or not isinstance(basket, list):
            return Response(
                {"error": "user_id and basket (list) are required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        recommendations = recommend_for_user(user_id, basket, top_k)
        return Response({"recommendations": recommendations})

    except Exception as e:
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)