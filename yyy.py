elif action_type == "BUTTON":

    if "name =" not in step_name:
        raise RuntimeError("BUTTON step missing button name")

    button_name = step_name.split("name =")[-1].strip().lower()
    step_lower = step_name.lower()

    nav_frame, content_frame = resolve_ccs_frames(page)

    # --------------------------------------------------
    # SECTION-SCOPED BUTTON RESOLUTION
    # --------------------------------------------------

    if "account activity" in step_lower:
        buttons = content_frame.locator(
            "div:has-text('Account Activity') "
            "input[type='button'], "
            "div:has-text('Account Activity') "
            "input[type='submit'], "
            "div:has-text('Account Activity') button"
        )

    elif "customer search" in step_lower:
        buttons = content_frame.locator(
            "div:has-text('Customer Search') "
            "input[type='button'], "
            "div:has-text('Customer Search') "
            "input[type='submit'], "
            "div:has-text('Customer Search') button"
        )

    else:
        # legacy fallback
        buttons = content_frame.locator(
            "input[type='button'], input[type='submit'], button"
        )

    count = await buttons.count()
    if count == 0:
        raise RuntimeError("No buttons found in scoped section")

    clicked = False

    for i in range(count):
        btn = buttons.nth(i)

        value_attr = (await btn.get_attribute("value") or "").lower()
        id_attr = (await btn.get_attribute("id") or "").lower()
        text_attr = (await btn.text_content() or "").lower()

        if button_name in value_attr or button_name in id_attr or button_name in text_attr:
            await btn.scroll_into_view_if_needed()
            await btn.wait_for(state="visible", timeout=10000)
            await btn.click()

            logger.info(
                LogCategory.EXECUTION,
                f"[PHASE 3] BUTTON click successful: {button_name}"
            )

            clicked = True
            break

    if not clicked:
        raise RuntimeError(f"Button '{button_name}' not found in specified section")
