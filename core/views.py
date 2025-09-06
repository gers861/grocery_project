import logging
from rest_framework import viewsets, status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import (
    User, Product, Cart, CartItem, Order, OrderItem,
    RecommendationLog, BehaviorLog
)
from .serializers import (
    UserSerializer, ProductSerializer, CartSerializer, CartItemSerializer,
    OrderSerializer, OrderItemSerializer,
    RecommendationLogSerializer, BehaviorLogSerializer
)

from .ml_models.loader import recommend_for_user, load_resources

# Load ML resources once when the server starts
load_resources()

# Logger
logger = logging.getLogger(__name__)

# --------------------------
# CRUD ViewSets for models
# --------------------------
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


# --------------------------
# AI Recommendations Endpoint
# --------------------------


def _normalize_item(s: str) -> str:
    return (s or "").strip().lower()

@api_view(["POST"])
def ml_recommendations(request):
    """
    POST body:
    {
        "user_id": 1808,          # number or string OK
        "basket": ["whole milk", "rolls/buns"],  # or [" Whole Milk ", "ROLLS/BUNS"]
        "top_k": 5
    }
    """
    try:
        user_id_raw = request.data.get("user_id")
        if user_id_raw is None:
            return Response({"error": "user_id is required."}, status=status.HTTP_400_BAD_REQUEST)

        # ✅ Make sure user_id is int so we hit the known-user path
        try:
            user_id = int(user_id_raw)
        except (TypeError, ValueError):
            return Response({"error": "user_id must be an integer."}, status=status.HTTP_400_BAD_REQUEST)

        basket_raw = request.data.get("basket", [])
        if isinstance(basket_raw, str):
            # accept a comma-separated string just in case
            basket_raw = [x for x in basket_raw.split(",") if x.strip()]

        if not isinstance(basket_raw, list):
            return Response({"error": "basket must be a list of item strings."}, status=status.HTTP_400_BAD_REQUEST)

        # ✅ Backend normalization so frontend casing/spacing doesn’t matter
        basket = [_normalize_item(x) for x in basket_raw]

        top_k = int(request.data.get("top_k", 5))
        logger.info("ML request -> user_id=%s, basket=%s, top_k=%s", user_id, basket, top_k)

        recs = recommend_for_user(user_id, basket, top_k)
        return Response({"recommendations": recs})

    except Exception as e:
        logger.exception("Recommendation failed")
        return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
