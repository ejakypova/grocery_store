import os
from datetime import datetime
from decimal import Decimal
from sqlalchemy import create_engine, Column, Integer, String, Numeric, DateTime, ForeignKey, UniqueConstraint, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, Session
from sqlalchemy.engine import URL
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")


class Base(DeclarativeBase): pass
class Store(Base):
    __tablename__ = 'stores'
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
class Product(Base):
    __tablename__ = 'products'
    id: Mapped[int] = mapped_column(primary_key=True)
    store_id: Mapped[int] = mapped_column(ForeignKey('stores.id'))
    name: Mapped[str] = mapped_column(String(600))
    url: Mapped[str] = mapped_column(String(2000))
    characteristics: Mapped[str | None] = mapped_column(String(1000))
    quantity: Mapped[Decimal | None] = mapped_column(Numeric(12, 3))
    unit: Mapped[str | None] = mapped_column(String(30))
    __table_args__ = (UniqueConstraint('store_id', 'url', 'name', name='uq_store_url_name'),)
class PriceHistory(Base):
    __tablename__ = 'price_history'
    id: Mapped[int] = mapped_column(primary_key=True)
    product_id: Mapped[int] = mapped_column(ForeignKey('products.id'), index=True)
    price_som: Mapped[Decimal] = mapped_column(Numeric(12, 2))
    old_price_som: Mapped[Decimal | None] = mapped_column(Numeric(12, 2))
    collected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    status: Mapped[str | None] = mapped_column(String(100))

def save_rows(rows):
    password = os.getenv("DB_PASSWORD")

    if password is None:
        raise RuntimeError("В .env не указан DB_PASSWORD")

    url = URL.create(
        drivername="postgresql+psycopg",
        username=os.getenv("DB_USER", "postgres"),
        password=password,
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", "5432")),
        database=os.getenv("DB_NAME", "grocery_ds"),
    )

    engine = create_engine(url, pool_pre_ping=True)

    try:
        Base.metadata.create_all(engine)

        with Session(engine) as session:
            for row in rows:
                store = session.scalar(
                    select(Store).where(
                        Store.name == row["store"]
                    )
                )

                if store is None:
                    store = Store(name=row["store"])
                    session.add(store)
                    session.flush()

                product = session.scalar(
                    select(Product).where(
                        Product.store_id == store.id,
                        Product.url == row["url"],
                        Product.name == row["name"],
                    )
                )

                if product is None:
                    product = Product(
                        store_id=store.id,
                        name=row["name"],
                        url=row["url"],
                    )
                    session.add(product)
                    session.flush()

                product.characteristics = row.get("characteristics")
                product.quantity = row.get("quantity")
                product.unit = row.get("unit")

                session.add(
                    PriceHistory(
                        product_id=product.id,
                        price_som=row["price_som"],
                        old_price_som=row.get("old_price_som"),
                        collected_at=datetime.fromisoformat(
                            row["collected_at"]
                        ).astimezone(),
                        status=row.get("status"),
                    )
                )

            session.commit()

    finally:
        engine.dispose()

    return len(rows)
