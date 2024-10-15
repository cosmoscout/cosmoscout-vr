export default function formatNumberWithColor(number, element) {
  if (number < 0) {
    element.classList.add("negative");
  } else {
    element.classList.remove("negative")
  }

  element.innerHTML = number.toLocaleString();
}