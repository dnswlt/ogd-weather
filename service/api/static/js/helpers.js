// safeUpdateYearInputs updates the form <input> fields from_year and to_year
window.safeUpdateYearInputs = function(firstYear, lastYear, minYear, maxYear) {
  const fy = document.getElementById('from_year');
  const ty = document.getElementById('to_year');
  if (!fy || !ty) return;

  fy.value = firstYear;
  ty.value = lastYear;
  fy.min = minYear;
  ty.min = minYear;
  fy.max = maxYear;
  ty.max = maxYear;
};
