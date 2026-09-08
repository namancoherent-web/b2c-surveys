import random
import re
import json
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

SOURCE_FILE = Path(
    r"C:\CI BB\b2c-survey-agent-main\output\global-running-shoes-market_canada.json"
)

OUTPUT_FOLDER = Path(
    r"D:\seed files b2c"
)

NUMBER_OF_FILES = 5000


# ============================================================
# SAME-LENGTH NUMBER GENERATOR
# ============================================================

def random_same_length_number(original):
    """
    Replace a numeric value with another numeric value having
    exactly the same number of characters.

    This is important because the generated JSON must remain
    exactly the same byte size as the original.
    """

    # Integer, e.g.:
    # 400 -> 731
    if re.fullmatch(r"\d+", original):
        length = len(original)

        if length == 1:
            return str(random.randint(0, 9))

        first = str(random.randint(1, 9))

        rest = "".join(
            str(random.randint(0, 9))
            for _ in range(length - 1)
        )

        return first + rest

    # Decimal, e.g.:
    # 38.0 -> 72.4
    # 5.0  -> 8.7
    if re.fullmatch(r"\d+\.\d+", original):

        integer_part, decimal_part = original.split(".")

        integer_length = len(integer_part)
        decimal_length = len(decimal_part)

        if integer_length == 1:
            new_integer = str(random.randint(0, 9))
        else:
            new_integer = (
                str(random.randint(1, 9))
                + "".join(
                    str(random.randint(0, 9))
                    for _ in range(integer_length - 1)
                )
            )

        new_decimal = "".join(
            str(random.randint(0, 9))
            for _ in range(decimal_length)
        )

        return new_integer + "." + new_decimal

    return original


# ============================================================
# CREATE DATA VARIANT
# ============================================================

def generate_variant(source_bytes):
    """
    Modify numeric seed data without changing:
      - JSON keys
      - arrays
      - strings
      - indentation
      - whitespace
      - number of objects
      - number of questions
      - JSON structure

    Every replacement is exactly the same character length.
    """

    text = source_bytes.decode("utf-8")

    # --------------------------------------------------------
    # Fields that represent actual data values in this file.
    #
    # Examples found in the uploaded file:
    #
    # "value": 38.0
    # "percentage": 35.0
    # "N": 400
    # "sample_size": 2500
    #
    # We intentionally do NOT modify qNum, funnel_position,
    # narrative_order, etc., because those describe structure.
    # --------------------------------------------------------

    data_fields = [
        "value",
        "percentage",
        "N",
        "sample_size",
    ]

    for field in data_fields:

        pattern = (
            rf'("{re.escape(field)}"\s*:\s*)'
            r'(\d+(?:\.\d+)?)'
        )

        def replace_value(match):
            prefix = match.group(1)
            original_number = match.group(2)

            new_number = random_same_length_number(
                original_number
            )

            return prefix + new_number

        text = re.sub(
            pattern,
            replace_value,
            text
        )

    return text.encode("utf-8")


# ============================================================
# VALIDATE JSON
# ============================================================

def validate_json(file_bytes):
    """
    Make sure generated data is valid JSON.
    """

    try:
        return json.loads(
            file_bytes.decode("utf-8")
        )
    except Exception as exc:
        raise RuntimeError(
            f"Generated file is not valid JSON:\n{exc}"
        )


# ============================================================
# GET STRUCTURE SIGNATURE
# ============================================================

def structure_signature(value):
    """
    Build a recursive signature of the JSON structure.

    Values themselves are ignored.

    This lets us verify that generated files have the same
    structure as the original.
    """

    if isinstance(value, dict):
        return (
            "dict",
            tuple(
                (
                    key,
                    structure_signature(child)
                )
                for key, child in value.items()
            )
        )

    if isinstance(value, list):
        return (
            "list",
            tuple(
                structure_signature(child)
                for child in value
            )
        )

    # Ignore actual primitive values.
    if isinstance(value, bool):
        return "bool"

    if isinstance(value, int):
        return "int"

    if isinstance(value, float):
        return "float"

    if isinstance(value, str):
        return "string"

    if value is None:
        return "null"

    return type(value).__name__


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("B2C RUNNING SHOES JSON SEED FILE GENERATOR")
    print("=" * 70)

    # --------------------------------------------------------
    # Check source
    # --------------------------------------------------------

    if not SOURCE_FILE.exists():

        raise FileNotFoundError(
            "\nSource file not found:\n"
            f"{SOURCE_FILE}\n\n"
            "Please verify that the path is correct."
        )

    # --------------------------------------------------------
    # Read original file as raw bytes
    # --------------------------------------------------------

    source_bytes = SOURCE_FILE.read_bytes()

    original_size = len(source_bytes)

    print()
    print(f"Source file:")
    print(SOURCE_FILE)

    print()
    print(
        f"Original file size: "
        f"{original_size:,} bytes"
    )

    # --------------------------------------------------------
    # Validate original JSON
    # --------------------------------------------------------

    original_data = validate_json(
        source_bytes
    )

    original_structure = structure_signature(
        original_data
    )

    print(
        "Original JSON: VALID"
    )

    # --------------------------------------------------------
    # Create output folder
    # --------------------------------------------------------

    OUTPUT_FOLDER.mkdir(
        parents=True,
        exist_ok=True
    )

    print()
    print(
        f"Output folder:\n{OUTPUT_FOLDER}"
    )

    print()
    print(
        f"Generating {NUMBER_OF_FILES:,} files..."
    )

    print()

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    successful = 0

    for i in range(
        1,
        NUMBER_OF_FILES + 1
    ):

        # Create variant.
        new_bytes = generate_variant(
            source_bytes
        )

        # ----------------------------------------------------
        # EXACT FILE SIZE CHECK
        # ----------------------------------------------------

        if len(new_bytes) != original_size:

            raise RuntimeError(
                "\nSIZE ERROR!\n"
                f"Original:  {original_size:,} bytes\n"
                f"Generated: {len(new_bytes):,} bytes\n"
                f"File: seed_{i:05d}.json"
            )

        # ----------------------------------------------------
        # Validate JSON
        # ----------------------------------------------------

        new_data = validate_json(
            new_bytes
        )

        # ----------------------------------------------------
        # STRUCTURE CHECK
        # ----------------------------------------------------

        new_structure = structure_signature(
            new_data
        )

        if new_structure != original_structure:

            raise RuntimeError(
                "\nSTRUCTURE ERROR!\n"
                f"Generated file: seed_{i:05d}.json"
            )

        # ----------------------------------------------------
        # Write
        # ----------------------------------------------------

        output_file = (
            OUTPUT_FOLDER
            / f"seed_{i:05d}.json"
        )

        output_file.write_bytes(
            new_bytes
        )

        # ----------------------------------------------------
        # Physical disk-size check
        # ----------------------------------------------------

        actual_size = output_file.stat().st_size

        if actual_size != original_size:

            raise RuntimeError(
                "\nDISK SIZE ERROR!\n"
                f"File: {output_file}\n"
                f"Expected: {original_size:,}\n"
                f"Actual: {actual_size:,}"
            )

        successful += 1

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        if (
            i % 100 == 0
            or i == NUMBER_OF_FILES
        ):

            percentage = (
                i / NUMBER_OF_FILES
            ) * 100

            print(
                f"\rProgress: "
                f"{i:,}/{NUMBER_OF_FILES:,} "
                f"({percentage:6.2f}%)",
                end="",
                flush=True
            )

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print()
    print()

    print("=" * 70)
    print("GENERATION COMPLETE")
    print("=" * 70)

    print(
        f"Source file       : "
        f"{SOURCE_FILE.name}"
    )

    print(
        f"Files generated   : "
        f"{successful:,}"
    )

    print(
        f"Size of each file : "
        f"{original_size:,} bytes"
    )

    total_gb = (
        original_size * successful
    ) / (1024 ** 3)

    print(
        f"Total data        : "
        f"{total_gb:.2f} GB"
    )

    print(
        f"Output folder     : "
        f"{OUTPUT_FOLDER}"
    )

    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()