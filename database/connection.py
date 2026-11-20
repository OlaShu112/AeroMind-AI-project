
import os
import mysql.connector
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "3306")),
        user=os.getenv("DB_USERNAME", "root"),
        password=os.getenv("DB_PASSWORD", ""),
        database=os.getenv("DB_DATABASE", "aeromind_ai"),
        connection_timeout=10
    )


def test_connection():
    try:
        connection = get_connection()

        cursor = connection.cursor()

        cursor.execute("SELECT DATABASE()")
        database_name = cursor.fetchone()[0]

        print("SUCCESS: AeroMind AI connected to MySQL")
        print(f"Database: {database_name}")

        cursor.execute("SHOW TABLES")
        tables = cursor.fetchall()

        print(f"Total tables: {len(tables)}")

        for table in tables:
            print(f" - {table[0]}")

        cursor.close()
        connection.close()

    except mysql.connector.Error as error:
        print(f"MySQL connection error: {error}")


if __name__ == "__main__":
    test_connection()
