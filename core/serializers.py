from rest_framework import serializers
from .models import (
    User, Product, Cart, CartItem, Order, OrderItem,
    RecommendationLog, BehaviorLog
)


class UserSerializer(serializers.ModelSerializer):
    class Meta:
        model = User
        fields = "__all__"


class ProductSerializer(serializers.ModelSerializer):
    class Meta:
        model = Product
        fields = "__all__"


class CartItemSerializer(serializers.ModelSerializer):
    product = ProductSerializer(read_only=True)
    product_id = serializers.PrimaryKeyRelatedField(
        queryset=Product.objects.all(), source="product", write_only=True
    )

    class Meta:
        model = CartItem
        fields = ("id", "cart", "product", "product_id", "quantity")


class CartSerializer(serializers.ModelSerializer):
    items = CartItemSerializer(many=True, read_only=True)

    class Meta:
        model = Cart
        fields = ("id", "user", "created_at", "items")


class OrderItemWriteSerializer(serializers.ModelSerializer):
    product = serializers.PrimaryKeyRelatedField(queryset=Product.objects.all())

    class Meta:
        model = OrderItem
        fields = ("product", "quantity", "price")


class OrderItemReadSerializer(serializers.ModelSerializer):
    product = ProductSerializer(read_only=True)

    class Meta:
        model = OrderItem
        fields = ("id", "product", "quantity", "price")


class OrderSerializer(serializers.ModelSerializer):
    # Use the correct related_name from OrderItem
    items = OrderItemReadSerializer(many=True, read_only=True)
    user = serializers.StringRelatedField(read_only=True)

    class Meta:
        model = Order
        fields = ["id", "user", "order_date", "total_amount", "items"]
        
        
    def create(self, validated_data):
        items_data = validated_data.pop("items", [])
        order = Order.objects.create(**validated_data)
        total = 0
        for item in items_data:
            prod = item["product"]
            qty = item.get("quantity", 1)
            price = prod.price
            OrderItem.objects.create(order=order, product=prod, quantity=qty, price=price)
            total += price * qty
        order.total_amount = total
        order.save()
        return order


class RecommendationLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = RecommendationLog
        fields = "__all__"


class BehaviorLogSerializer(serializers.ModelSerializer):
    class Meta:
        model = BehaviorLog
        fields = "__all__"
