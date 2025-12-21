When the user checks the checkbox for transaction with description = Zelle payment from Scott Snyder


elif "checks the checkbox" in step_lower or "selects the checkbox" in step_lower:
    action_type = "CHECKBOX"


elif action_type == "CHECKBOX":

    if "description =" not in step_name.lower():
        raise RuntimeError("CHECKBOX step missing transaction description")

    txn_desc = step_name.split("description =")[-1].strip().lower()

    nav_frame, content_frame = resolve_ccs_frames(page)

    # 1️⃣ Locate Account Activity table rows
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
        if txn_desc in row_text:
            checkbox = row.locator("input[type='checkbox']")

            if await checkbox.count() != 1:
                raise RuntimeError(
                    f"Expected 1 checkbox in matched row, found {await checkbox.count()}"
                )

            await checkbox.scroll_into_view_if_needed()
            await checkbox.wait_for(state="visible", timeout=10000)
            await checkbox.check(force=True)

            logger.info(
                LogCategory.EXECUTION,
                f"[PHASE 3] CHECKBOX selected for transaction: {txn_desc}"
            )

            matched = True
            break

    if not matched:
        raise RuntimeError(
            f"Transaction with description '{txn_desc}' not found"
        )

    logger.info(
        LogCategory.EXECUTION,
        "[PHASE 3] CHECKBOX selection successful"
    )
