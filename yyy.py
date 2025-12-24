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
    # 2. Resolve CCS frames
    # -------------------------------
    nav_frame, content_frame = resolve_ccs_frames(page)

    # -------------------------------
    # 3. Locate radio button (STRICT & EXACT)
    # -------------------------------
    radio = content_frame.locator(
        f"input[type='radio'][value='{radio_value}']"
    )

    if await radio.count() != 1:
        raise RuntimeError(
            f"Expected 1 radio button with value '{radio_value}', "
            f"found {await radio.count()}"
        )

    # -------------------------------
    # 4. Interact safely
    # -------------------------------
    await radio.scroll_into_view_if_needed()
    await radio.wait_for(state="visible", timeout=10000)

    if not await radio.is_checked():
        await radio.check(force=True)

    logger.info(
        LogCategory.EXECUTION,
        f"[PHASE 3] Radio button selected with value = {radio_value}"
    )
