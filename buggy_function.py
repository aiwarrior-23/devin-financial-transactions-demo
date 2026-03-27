def calculate_success_ratio(success_count, failure_count):
    total = success_count + failure_count

    if total == 0:
        ratio = 0.0
    else:
        ratio = round(success_count / total, 2)

    return {
        "total_requests": total,
        "success_ratio": ratio
    }
