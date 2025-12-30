elif action_type == "RADIO":

    step_lower = step_name.lower()

    # -------------------------------
    # 1. Extract radio value (Y / N)
    # -------------------------------
    match = re.search(r"value\s*=\s*([yn])", step_lower)
    if not match:
        raise RuntimeError("RADIO step must specify value = Y or N")

    radio_value = match.group(1).upper()

    # -------------------------------
    # 2. Resolve frames
    # -------------------------------
    nav_frame, content_frame = resolve_ccs_frames(page)

    # -------------------------------
    # 3. Locate radio via XPATH (STRICT)
    # -------------------------------
    radio = content_frame.locator(
        f"xpath=//input[@type='radio' and @value='{radio_value}']"
    ).first

    # -------------------------------
    # 4. Interact (NO SCROLL)
    # -------------------------------
    await radio.wait_for(state="attached", timeout=10000)

    if not await radio.is_checked():
        await radio.check(force=True)

    logger.info(
        LogCategory.EXECUTION,
        f"[PHASE 3] Radio button selected with value = {radio_value}"
    )
