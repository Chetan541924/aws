elif action_type == "RADIO":
    step_lower = step_name.lower()

    # Extract answer index (answer = 1 / 2)
    match = re.search(r"answer\s*=\s*(\d+)", step_lower)
    if not match:
        raise RuntimeError("RADIO step must specify answer = <number>")

    answer_index = match.group(1)

    # Resolve frames
    nav_frame, content_frame = resolve_ccs_frames(page)

    # ============================================================
    # CRITICAL: Wait for page to stabilize after previous actions
    # ============================================================
    await page.wait_for_timeout(3000)  # Let any auto-scroll finish
    await content_frame.wait_for_load_state("networkidle", timeout=15000)

    # Locate radio by ID suffix
    radio_selector = f"input[type='radio'][id$='answer{answer_index}']"
    radio = content_frame.locator(radio_selector)

    # Wait for radio to exist
    await radio.wait_for(state="attached", timeout=10000)

    if await radio.count() == 0:
        raise RuntimeError(
            f"Radio with id ending 'answer{answer_index}' not found"
        )

    radio = radio.first

    # ============================================================
    # FORCE scroll into view BEFORE clicking
    # ============================================================
    try:
        await radio.scroll_into_view_if_needed(timeout=5000)
    except Exception as e:
        logger.warning(
            LogCategory.EXECUTION,
            f"Scroll failed, attempting direct scroll: {str(e)}"
        )
        # Fallback: Force scroll using JavaScript
        await content_frame.evaluate(
            """(selector) => {
                const element = document.querySelector(selector);
                if (element) {
                    element.scrollIntoView({behavior: 'instant', block: 'center'});
                }
            }""",
            radio_selector
        )
        await page.wait_for_timeout(1000)

    # ============================================================
    # Verify visibility before clicking
    # ============================================================
    await radio.wait_for(state="visible", timeout=10000)

    # ============================================================
    # Click using JavaScript (most reliable for radio buttons)
    # ============================================================
    await content_frame.evaluate(
        """(selector) => {
            const element = document.querySelector(selector);
            if (element) {
                element.click();
            } else {
                throw new Error('Radio button not found for click');
            }
        }""",
        radio_selector
    )

    # Visual verification delay
    await page.wait_for_timeout(2000)

    # Verify radio is checked
    is_checked = await radio.is_checked()
    if not is_checked:
        raise RuntimeError(
            f"Radio button answer={answer_index} was clicked but not selected"
        )

    logger.info(
        LogCategory.EXECUTION,
        f"[PHASE 3] RADIO selected successfully: answer={answer_index}"
    )
