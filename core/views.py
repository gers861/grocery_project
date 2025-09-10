from rest_framework import viewsets, status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.contrib.auth import get_user_model
from rest_framework.decorators import action
from django.shortcuts import get_object_or_404

from .models import (
    User, Product, Cart, CartItem, Order, OrderItem,
    RecommendationLog, BehaviorLog
)
from .serializers import (
    UserSerializer, ProductSerializer, CartSerializer, CartItemSerializer,
    OrderSerializer, OrderItemReadSerializer,
    RecommendationLogSerializer, BehaviorLogSerializer
)

from .ml_models.loader import recommend_for_user, load_resources, user2idx
import logging
UserModel = get_user_model()


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

  

logger = logging.getLogger(__name__)


class RecommendationLogViewSet(viewsets.ModelViewSet):
    queryset = RecommendationLog.objects.all()
    serializer_class = RecommendationLogSerializer


class BehaviorLogViewSet(viewsets.ModelViewSet):
    queryset = BehaviorLog.objects.all()
    serializer_class = BehaviorLogSerializer

@api_view(["POST"])
def checkout_cart(request, cart_id):
    """
    Convert a Cart into an Order, generate recommendations,
    clear the cart, and return recommendations.
    """
    try:
        cart = get_object_or_404(Cart, id=cart_id)
        
        # 🚨 Check if cart is empty
        if not cart.items.exists():
            return Response(
                {"error": "Cart is empty. Add products before checkout."},
                status=status.HTTP_400_BAD_REQUEST
            )

        # 1. Create a new Order
        order = Order.objects.create(
            user=cart.user,
            total_amount=sum(
            item.product.price * item.quantity for item in cart.items.all()
            )
        )

        # 2. Create OrderItems
        for item in cart.items.all():
            OrderItem.objects.create(
                order=order,
                product=item.product,
                quantity=item.quantity,
                price=item.product.price,
            )

        # 3. Prepare basket for recommendations
        basket = [item.product.name.lower() for item in cart.items.all()]

        recommendations = recommend_for_user(cart.user.id, basket, top_k=5)

        # 4. Save RecommendationLog
        RecommendationLog.objects.create(
            user=cart.user,
            order=order,
            recommendations=recommendations,
            source="ml"
        )

        # 5. Clear cart
        cart.items.all().delete()

        return Response(
            {"order_id": order.id, "recommendations": recommendations},
            status=status.HTTP_201_CREATED,
        )

    except Exception as e:
        return Response(
            {"error": f"Checkout failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )


@api_view(["POST"])
def ml_recommendations(request):
    try:
        user_id = request.data.get("user_id")
        basket_raw = request.data.get("basket", [])
        basket = [item.strip().lower() for item in basket_raw]
        top_k = int(request.data.get("top_k", 5))

        if not user_id:
            return Response({"error": "user_id is required."}, status=status.HTTP_400_BAD_REQUEST)

        recs = recommend_for_user(user_id, basket, top_k)
        return Response({"recommendations": recs})

    except Exception as e:
        return Response(
            {"error": f"Recommendation failed: {str(e)}"},
            status=status.HTTP_500_INTERNAL_SERVER_ERROR,
        )

