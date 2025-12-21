elif action_type == "BUTTON":

    if "name =" not in step_name:
        raise RuntimeError("BUTTON step missing button name")

    step_lower = step_name.lower()
    button_name = step_name.split("name =")[-1].strip().lower()

    nav_frame, content_frame = resolve_ccs_frames(page)

    # -------------------------------
    # Resolve section explicitly
    # -------------------------------
    section = None
    if " in the " in step_lower:
        section = (
            step_lower.split(" in the ", 1)[1]
            .split(" with name", 1)[0]
            .replace(" section", "")
            .strip()
        )

    if section:
        section_root = content_frame.locator(
            f"div:has-text('{section.title()}')"
        )
        buttons = section_root.locator(
            "input[type='button'], input[type='submit'], button"
        )
    else:
        buttons = content_frame.locator(
            "input[type='button'], input[type='submit'], button"
        )

    count = await buttons.count()
    if count == 0:
        raise RuntimeError(f"No buttons found in section: {section}")

    for i in range(count):
        btn = buttons.nth(i)

        value = (await btn.get_attribute("value") or "").lower()
        text = (await btn.text_content() or "").lower()
        id_ = (await btn.get_attribute("id") or "").lower()

        if button_name in (value + text + id_):
            await btn.scroll_into_view_if_needed()
            await btn.hover()
            await content_frame.wait_for_timeout(200)
            await btn.click(force=True)

            logger.info(
                LogCategory.EXECUTION,
                f"[PHASE 3] BUTTON click successful: {button_name} in {section}"
            )
            return

    raise RuntimeError(
        f"Button '{button_name}' not found in section '{section}'"
    )
