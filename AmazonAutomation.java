import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.interactions.Actions;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;

import java.time.Duration;
import java.util.List;

public class AmazonAutomation {

    private WebDriver driver;
    private WebDriverWait wait;

    public AmazonAutomation(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, Duration.ofSeconds(10));
    }

    // ------------------------------------------------------------
    // LOGIN + BASIC NAVIGATION
    // ------------------------------------------------------------

    public void navigateToHomepage() {
        driver.get("https://www.amazon.in");
    }

    public void clickSignInButton() {
        wait.until(ExpectedConditions.elementToBeClickable(By.id("nav-link-accountList"))).click();
    }

    public void enterEmail(String email) {
        WebElement field = wait.until(ExpectedConditions.presenceOfElementLocated(By.id("ap_email")));
        field.clear();
        field.sendKeys(email);
    }

    public void clickContinueAfterEmail() {
        wait.until(ExpectedConditions.elementToBeClickable(By.id("continue"))).click();
    }

    public void enterPassword(String password) {
        WebElement field = wait.until(ExpectedConditions.presenceOfElementLocated(By.id("ap_password")));
        field.clear();
        field.sendKeys(password);
    }

    public void submitLogin() {
        wait.until(ExpectedConditions.elementToBeClickable(By.id("signInSubmit"))).click();
    }

    public void login(String email, String password) {
        navigateToHomepage();
        clickSignInButton();
        enterEmail(email);
        clickContinueAfterEmail();
        enterPassword(password);
        submitLogin();
    }

    // ------------------------------------------------------------
    // SEARCH + PRODUCT DETAILS
    // ------------------------------------------------------------

    public void searchProduct(String productName) {
        WebElement searchBox = wait.until(
                ExpectedConditions.presenceOfElementLocated(By.id("twotabsearchtextbox"))
        );
        searchBox.clear();
        searchBox.sendKeys(productName);
        searchBox.submit();
    }

    public void applyPriceFilter(int minPrice, int maxPrice) {
        WebElement minField = wait.until(
                ExpectedConditions.presenceOfElementLocated(By.id("low-price"))
        );
        WebElement maxField = wait.until(
                ExpectedConditions.presenceOfElementLocated(By.id("high-price"))
        );

        minField.sendKeys(String.valueOf(minPrice));
        maxField.sendKeys(String.valueOf(maxPrice));

        driver.findElement(By.xpath("//input[@class='a-button-input']")).click();
    }

    public void openFirstSearchResult() {
        WebElement product = wait.until(
                ExpectedConditions.elementToBeClickable(
                        By.xpath("//div[@data-cel-widget='search_result_1']//h2/a")
                )
        );
        product.click();
    }

    // ------------------------------------------------------------
    // CART METHODS
    // ------------------------------------------------------------

    public void addToCart() {
        WebElement btn = wait.until(
                ExpectedConditions.elementToBeClickable(By.id("add-to-cart-button"))
        );
        btn.click();
    }

    public void openCart() {
        wait.until(ExpectedConditions.elementToBeClickable(By.id("nav-cart"))).click();
    }

    public void removeFirstCartItem() {
        WebElement deleteBtn = wait.until(
                ExpectedConditions.elementToBeClickable(By.xpath("//input[@value='Delete']"))
        );
        deleteBtn.click();
    }

    public int getCartItemCount() {
        WebElement countEl = wait.until(
                ExpectedConditions.presenceOfElementLocated(By.id("nav-cart-count"))
        );
        return Integer.parseInt(countEl.getText().trim());
    }

    public void proceedToCheckout() {
        WebElement btn = wait.until(
                ExpectedConditions.elementToBeClickable(By.name("proceedToRetailCheckout"))
        );
        btn.click();
    }

    public boolean validateCartHasItems() {
        return getCartItemCount() > 0;
    }

    // ------------------------------------------------------------
    // DELIVERY + CHECKOUT FLOW
    // ------------------------------------------------------------

    public void chooseDeliveryAddress() {
        WebElement addr = wait.until(
                ExpectedConditions.elementToBeClickable(
                        By.xpath("//div[@id='address-book-entry-0']//a")
                )
        );
        addr.click();
    }

    public void chooseDeliveryOption() {
        WebElement opt = wait.until(
                ExpectedConditions.elementToBeClickable(
                        By.name("ppw-widgetEvent:SetPaymentPlanSelectContinueEvent")
                )
        );
        opt.click();
    }

    public void placeOrder() {
        WebElement place = wait.until(
                ExpectedConditions.elementToBeClickable(By.name("placeYourOrder1"))
        );
        place.click();
    }

    // ------------------------------------------------------------
    // ORDERS FLOW
    // ------------------------------------------------------------

    public void navigateToOrders() {
        wait.until(ExpectedConditions.elementToBeClickable(By.id("nav-orders"))).click();
    }

    public void openFirstOrder() {
        WebElement order = wait.until(
                ExpectedConditions.elementToBeClickable(
                        By.xpath("//div[contains(@class,'order')]//a")
                )
        );
        order.click();
    }

    // ------------------------------------------------------------
    // WISHLIST METHODS
    // ------------------------------------------------------------

    public void navigateToWishlist() {
        wait.until(ExpectedConditions.elementToBeClickable(By.id("nav-wishlist"))).click();
    }

    public void addCurrentItemToWishlist() {
        WebElement btn = wait.until(
                ExpectedConditions.elementToBeClickable(By.id("add-to-wishlist-button-submit"))
        );
        btn.click();
    }

    public void selectDefaultWishlist() {
        WebElement wl = wait.until(
                ExpectedConditions.elementToBeClickable(By.xpath("//span[text()='Wish List']"))
        );
        wl.click();
    }

    public void openWishlist() {
        driver.get("https://www.amazon.in/hz/wishlist/ls");
    }

    public void removeWishlistItem() {
        WebElement remove = wait.until(
                ExpectedConditions.elementToBeClickable(By.name("submit.deleteItem"))
        );
        remove.click();
    }

    public boolean validateWishlistHasItems() {
        List<WebElement> items = wait.until(
                ExpectedConditions.presenceOfAllElementsLocatedBy(
                        By.className("g-item-sortable")
                )
        );
        return items.size() > 0;
    }

    // ------------------------------------------------------------
    // USER ADDRESS MANAGEMENT
    // ------------------------------------------------------------

    public void navigateToAddresses() {
        driver.get("https://www.amazon.in/a/addresses");
    }

    public void addNewAddress(String name, String phone, String pincode, String line1) {
        wait.until(ExpectedConditions.elementToBeClickable(
                By.id("ya-myab-address-add-link"))).click();

        wait.until(ExpectedConditions.presenceOfElementLocated(
                By.id("address-ui-widgets-enterAddressFullName"))).sendKeys(name);

        driver.findElement(By.id("address-ui-widgets-enterAddressPhoneNumber")).sendKeys(phone);
        driver.findElement(By.id("address-ui-widgets-enterAddressPostalCode")).sendKeys(pincode);
        driver.findElement(By.id("address-ui-widgets-enterAddressLine1")).sendKeys(line1);

        driver.findElement(By.id("address-ui-widgets-form-submit-button")).click();
    }

    public void deleteFirstAddress() {
        WebElement deleteBtn = wait.until(
                ExpectedConditions.elementToBeClickable(
                        By.xpath("//a[contains(@href, 'delete')]")
                )
        );
        deleteBtn.click();

        WebElement confirm = wait.until(
                ExpectedConditions.elementToBeClickable(By.id("deleteAddressModal-announce"))
        );
        confirm.click();
    }

    // ------------------------------------------------------------
    // LOGOUT
    // ------------------------------------------------------------

    public void logoutOpenMenu() {
        WebElement hover = wait.until(
                ExpectedConditions.presenceOfElementLocated(By.id("nav-link-accountList"))
        );
        Actions actions = new Actions(driver);
        actions.moveToElement(hover).perform();
    }

    public void logout() {
        logoutOpenMenu();
        WebElement signout = wait.until(
                ExpectedConditions.elementToBeClickable(
                        By.xpath("//span[text()='Sign Out']")
                )
        );
        signout.click();
    }

    public boolean validateLoggedIn() {
        WebElement profile = wait.until(
                ExpectedConditions.presenceOfElementLocated(
                        By.id("nav-link-accountList-nav-line-1")
                )
        );
        return profile.getText().contains("Hello");
    }

    // ------------------------------------------------------------
    // UTILITIES
    // ------------------------------------------------------------

    public void navigateBack() {
        driver.navigate().back();
    }

    public void refreshPage() {
        driver.navigate().refresh();
    }
}
