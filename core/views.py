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

@api_view(["POST"])
def checkout(request, cart_id):
    """
    Convert Cart → Order, then fetch AI recommendations
    """
    try:
        cart = Cart.objects.prefetch_related("items__product").get(id=cart_id)
        user = cart.user

        # Create Order
        order = Order.objects.create(
            user=user,
            total_amount=sum(item.product.price * item.quantity for item in cart.items.all())
        )

        # Create OrderItems
        for item in cart.items.all():
            OrderItem.objects.create(
                order=order,
                product=item.product,
                quantity=item.quantity,
                price=item.product.price,
            )

        # Build basket for ML
        basket = [item.product.name for item in cart.items.all()]
        recos = recommend_for_user(user.id, basket, top_k=5)

        # Save reco log
        RecommendationLog.objects.create(
            user=user,
            order=order,
            recommendations=recos,
            source="ml",
        )

        # Clear cart after checkout
        cart.items.all().delete()

        return Response(
            {"order_id": order.id, "recommendations": recos},
            status=status.HTTP_201_CREATED
        )

    except Cart.DoesNotExist:
        return Response({"error": "Cart not found"}, status=status.HTTP_404_NOT_FOUND)
    except Exception as e:
        logger.error("Checkout failed: %s", str(e))
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
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

