export default function(instance) {
  instance.registerHelper('lookup', function(obj, field) {
    if (!obj) {
      return obj;
    }
    if (String(field) === 'constructor' && !obj.propertyIsEnumerable(field)) {
      return undefined;
    }
    return obj[field];
  });
}
