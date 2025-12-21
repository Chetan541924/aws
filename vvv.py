When the user checks the checkbox for transaction with description = Zelle payment from Scott Snyder


elif "checks the checkbox" in step_lower or "selects the checkbox" in step_lower:
    action_type = "CHECKBOX"



elif action_type == "CHECKBOX":

    step_lower = step_name.lower()
    nav_frame, content_frame = resolve_ccs_frames(page)

    # Scope to Account Activity table
    rows = content_frame.locator(
        "div:has-text('Account Activity') tr"
    )

    row_count = await rows.count()
    if row_count == 0:
        raise RuntimeError("No rows found in Account Activity table")

    matched = False

    for i in range(row_count):
        row = rows.nth(i)

        row_text = (await row.text_content() or "").lower()

        if "zelle payment from scott snyder" in row_text:
            checkbox = row.locator("input[type='checkbox']")

            await checkbox.scroll_into_view_if_needed()
            await checkbox.wait_for(state="visible", timeout=10000)

            if not await checkbox.is_checked():
                await checkbox.check(force=True)

            matched = True
            break

    if not matched:
        raise RuntimeError("Matching checkbox row not found")

    # Allow UI to update
    await page.wait_for_timeout(500)

    logger.info(
        LogCategory.EXECUTION,
        "[PHASE 3] CHECKBOX selection successful"
    )
